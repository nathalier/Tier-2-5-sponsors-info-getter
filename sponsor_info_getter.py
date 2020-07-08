import time
import requests
import re
import pandas as pd
import Levenshtein

app_name = 'sqlhub'
api_key = ''
search_api_url = 'https://api.companieshouse.gov.uk/search/companies/'
get_info_api_url = 'https://api.companieshouse.gov.uk/company/' #{company_number}' ; 401 possible
sic_codes = pd.read_csv('SIC_cond_list_en.csv', index_col='SIC Code')
# COMPANIES_LIMIT = number of companies to be watched in search results.
# Needed company, if found, is usually within the first 3 (in 90% of cases) results
COMPANIES_LIMIT = 5
# BATCH_SIZE is used to periodically write proceeded results to a file
BATCH_SIZE = 500

period_start_time, req_num, states = -1, 0, set()
tot, not_obv, not_found = 0, 0, 0
unambig_results = []
ambig_results = []
not_found_results = []


def wait_for_slot(error=False):
    global req_num, period_start_time
    if period_start_time == -1:
        period_start_time = time.perf_counter()
    if req_num >= 595 or error:
        time.sleep(max(0, 310 - (time.perf_counter() - period_start_time)))
        req_num = 0
        period_start_time = time.perf_counter()


def search_request(search_name):
    global req_num
    req_num += 1
    wait_for_slot()
    r = requests.get(f'{search_api_url}?q={search_name.replace("&", "%26")}&items_per_page={COMPANIES_LIMIT}',
                 auth=(api_key, ''))
    return r


def get_info_request(c_code):
    global req_num
    req_num += 1
    wait_for_slot()
    r = requests.get(f'{get_info_api_url}{c_code}', auth=(api_key, ''))
    return r


def addr_to_str(addr):
    addr = addr or {}
    return ', '.join(filter(None, [addr.get('premises', ''), addr.get('address_line_1', ''), addr.get('address_line_2', ''),
                      addr.get('address_line_3', ''), addr.get('locality', ''), addr.get('region', ''),
                      addr.get('country', ''), addr.get('postal_code', '')]))


def add_info(companies, confidence='0'):
    """Makes another API request and dds the registration address and
    sic-codes list with the description (nature of business) to the company info"""
    for c in companies:
        if not c.get('company_number'):
            continue
        r = get_info_request(c.get('company_number', ''))
        if r.status_code != 200:
            continue
        c_info = r.json()
        c["reg_address"] = c_info.get("address_snippet", '') or \
                                         addr_to_str(c_info.get("registered_office_address", {})) or \
                                         addr_to_str(c_info.get("address", {}))
        c["business_activity"] = c_info.get("branch_company_details", {}).get("business_activity", '') or \
                                 c_info.get("business_activity", '')
        c["sic_codes"] = c_info.get("sic_codes", [])
        c["sic_description"] = '/ '.join(filter(None, [sic_codes['Description'].get(int(x), f'sic_code {x} not found')\
                                                       for x in c["sic_codes"]])) or c["business_activity"]
        c['confidence'] = confidence
    return companies


def common_address_lines(company, spons_address):
    """Checks whether the given company and sponsor's company have common words in addresses """
    addr = company.get('address') or dict()
    full_address = {x.lower() for x in spons_address if x == x}  # filters out nan values
    comp_address = {x.lower() for x in [addr.get('locality', ''), addr.get('region', '')] +
                                        addr.get('address_line_1', '').split(', ') +
                                        addr.get('address_line_1', '').split(', ') if x != ''}
    return len(full_address.intersection(comp_address)) > 0


def is_active(company):
    global states
    states.add(company.get('status'))
    return company.get('status', '') in ['active', 'open']


def active_index(companies):
    """returns first company in a 'companies' list in 'active' or 'open' state"""
    for c in companies:
        if is_active(c):
            return c


def penalize(matched, min_d):
    """Calculates penalties to companies from the 'matched' list which describe
    their difference from the sponsor being searched for"""
    penalties = []
    for comp in matched:
        penalty = 0
        penalty += comp['order_in_search'] // 2
        penalty += (comp['l_dist'] - min_d) // 2
        penalty += 1 - comp['dist_from_name']
        penalty += 1 - comp['address_match']
        penalty += 1 - (comp['simp_comp_name'].startswith(comp['name_shorten']) or \
                        comp['simp_comp_name'] in comp['simp_sponsor_name'])  # TODO to change
        penalty += 1 - is_active(comp)
        penalties.append(penalty)
    return penalties


def break_tie(best, penalties):
    """Chooses the best matching company from the final list of the most closely matched"""
    if 'BR' in {x['company_number'][:2] for x in best}:
        br_index = [i for i, x in enumerate(best) if x['company_number'][:2] == 'BR'][0]
        if penalties[br_index] - min(penalties) <= 1: # then probably that company is the best choice
            if [x['simp_comp_name'] for x in best].count(best[br_index]['simp_comp_name']) > 1:
                if 'FC' in {x['company_number'][:2] for x in best}:
                    fc_index = [i for i, x in enumerate(best) if x['company_number'][:2] == 'FC'][0]
                    if best[br_index]['simp_comp_name'] == best[fc_index]['simp_comp_name']:
                        best[br_index]['sic_description'] = best[br_index]['sic_description'] or\
                                                            best[br_index]['business_activity']
                        # in this case FC is a better choice between the two firms
                        del best[fc_index]
                        del penalties[fc_index]
                else:  # the firm with the same name without FC in name is a better choice
                    del best[br_index]
                    del penalties[br_index]

    if len(best) > 1:
        # if only one is active - select that one
        if sum([is_active(x) for x in best]) == 1:
            best = [active_index(best)]
        # if only one of them has matching address - select that one
        elif sum([x['address_match'] for x in best]) == 1:
            ind = [x['address_match'] for x in best].index(True)
            best = [best[ind]]
        # otherwise choose by min penalties and position
        else:
            first_min_ind = penalties.index(min(penalties))
            best = [best[first_min_ind]]

    return best


def best_match(companies):
    ambiguity = False
    penalties = []
    confidence = 0
    if len(companies) == 0:
        return companies, ambiguity, penalties

    distances = [x['l_dist'] for x in companies]
    min_dist = min(distances)
    num_min_dist = sum([x == min_dist for x in distances])

    # fastpath conditions (to return first found entry): first entry has min l_dist, it's <= 3 and is unique,
    #          company status is active(open) and address matches (or there are no results with matching address)
    if companies[0]['l_dist'] == min_dist <= 3 and num_min_dist == 1 and is_active(companies[0]) and \
                    (companies[0]['address_match'] or not any([x['address_match'] for x in companies])):
        return add_info([companies[0]], confidence=0.95), ambiguity, penalties

    # there is usually a variation in company name ending. So 30% shorter name is used
    filtered_companies = [x for x in companies
                          if x['simp_comp_name'].startswith(x['name_shorten'])]
    best = filtered_companies if min_dist <= 6 else []
    if best:
        confidence = 0.85

    if min_dist <= 8 and not best:
        best = [x for x in companies if (x['l_dist'] <= (min_dist + 5) and x['address_match'] or
                                         x['l_dist'] <= (min_dist + 3) and not x['address_match'])
                                    and (x['name_shorten'] in x['simp_comp_name'] or
                                         x['simp_sponsor_name'][:-1] in x['simp_comp_name'])]
        confidence = 0.5

    # if still no companies were selected we assume the result is too ambiguous
    if not best:
        ambiguity = True
        best, confidence = companies, 0.1

    if len(best) > 1:
        penalties = penalize(best, min_dist)
        min_penalty = min(penalties)

    if not ambiguity:  # nothing to do with too ambiguous results
        if penalties:  # that is, there are several results
            if min_penalty <= 6:
                two_min_pen = sorted(set(penalties))[:2]
                selected = [(x, penalties[i]) for i, x in enumerate(best) if penalties[i] in two_min_pen]
                best = break_tie([x[0] for x in selected], [x[1] for x in selected])
            else:
                ambiguity = True

    # if there is only one search result (even not very similar) we still add it
    # the level of confidence will be 0.1 for those
    if len(best) == 1:
        best = add_info(best, confidence)

    return best, ambiguity, penalties


def simplify(name):
    # substitution will be perforemed in the defined way so 'limited' word not to be substituted too early
    abbreviations = [('PUBLICLIMITEDCOMPANY', 'PLC'), ('LIMITEDLIABILITYPARTNERSHIPS', 'LLP'),
                     ('LIMITEDLIABILITYPARTNERSHIP', 'LLP'), ('LIMITED', 'LTD'),
                     ('TRUSTEES', 'T'), ('TRUSTEE', 'T'), ('STUDIOS', 'S'),
                     ('TRADING', 'T'), ('PRACTICE', 'P'), ('UKBRANCH', 'UK'), ('LONDONBRANCH', 'LO'),
                     ('SCOTLAND', 'SC'), ('ENGLAND', 'EN'), ('UNITEDKINGDOM', 'UK'),
                     ('CIC', '$'), ('CIO', '$'), ('RTM', '$'), ('INCORPORATED', 'INC'),
                     ('INTERNATIONAL', 'INT')]
    name = name.upper()
    name = re.sub(r'\bAND\b', '&', name, flags=re.IGNORECASE)
    name = re.sub('^THE ', '', name, flags=re.IGNORECASE)
    name = re.sub('^A ', '', name, flags=re.IGNORECASE)
    name = re.sub('(\.|,|;|:|-| |&|\'|"|\(|\))*', '', name)
    for x in abbreviations:
        name = re.sub(x[0], x[1], name)
    # LTD is often missed from companies name, so to decrease difference it's substituted with only 1 character
    name = re.sub('LTD', '~', name)
    name = re.sub('INC', '^', name)
    # name = re.sub(f'{"|".join(terms_to_shorten)}', '*', name)
    return name.strip()


def search_company(sponsor):
    global tot, not_obv, not_found, unambig_results, ambig_results, not_found_results
    tot += 1
    search_name = sponsor["name"].upper()

    # some sponsor names contain two variants of naming separated by 't/a', 't/as' etc.
    if match := re.search(r'(.*)(\bas\b|\bt/a\b|\bta\b|\bt/as\b)', search_name, flags=re.IGNORECASE):
        search_name = match.group(1)

    resp = search_request(search_name)
    # it shouldn't happen but just in case..
    while resp.status_code == 429:
        wait_for_slot(error=True)
        resp = search_request(search_name)
    if resp.status_code != 200:
        print(f'JSON Error!! {sponsor["name"]}')
        return
    search_result = resp.json()
    search_name = simplify(search_name)

    matching_companies = []
    for ind, comp in enumerate(search_result.get('items', [])):
        address_match = common_address_lines(comp, sponsor[['city', 'county']])
        name = simplify(comp.get('title', ''))
        snippet = comp.get('snippet', '') or name
        snippet = simplify(snippet)
        distances = [Levenshtein.distance(search_name, name), Levenshtein.distance(search_name, snippet)]
        min_dist = min(distances)
        c_profile = {
                     'sponsor_name': sponsor['name'],
                     'found_name': f"{comp.get('title', '')} | {comp.get('snippet', '')}",
                     'l_dist': min_dist,
                     'dist_from_name': distances[0] == min_dist, # min_dist got from name, not snippet
                     'address_match': address_match,
                     'order_in_search': ind,
                     'simp_sponsor_name': search_name,
                     'simp_comp_name': name if distances[0] == min_dist else snippet,
                     'status': comp.get("company_status", ''),
                     'address': comp.get('address', dict()),
                     'company_number': comp.get('company_number', ''),
                     'reg_address': '',
                     'business_activity': '',
                     'sic_codes': [],
                     'sic_description': '',
                     'confidence': '',
                     }
        c_profile['name_shorten'] = c_profile['simp_sponsor_name']\
                                    [: -round(len(c_profile['simp_sponsor_name']) * 0.3)]
        matching_companies.append(c_profile)

    matching_companies, ambiguity, penalties = best_match(matching_companies)

    if not matching_companies:
        not_found += 1
        not_found_results.append(sponsor["name"])
    elif len(matching_companies) == 1:
        matching_companies[0]['city'] = sponsor['city']
        matching_companies[0]['county'] = sponsor['county']
        unambig_results.append(matching_companies[0])
        if ambiguity:
            not_obv += 1
            print(f'Results were ambiguous for {sponsor["name"]}')
    else:
        print(f'{sponsor["name"]} - Results len: {len(matching_companies)}. Ambiguity: {ambiguity}.')
        not_obv += 1
        ambig_results.append(sponsor["name"])
        for c in matching_companies:
            keys_to_print = list(c)
            keys_to_print.remove('address')  # makes the record unreadable
            print([c[key] for key in keys_to_print])
        print(penalties)


def append_unambig(res_columns, batch_num, file_name):
    with open(file_name, 'a', encoding='utf-8') as f:
        if batch_num == 0:
            f.write(';'.join(res_columns) + '\n')
        f.write('\n'.join([';'.join([str(x[col]) for col in res_columns]) for x in unambig_results]))
        f.write('\n')


def get_sponsors_info(file_name, encoding='utf-8'):
    global unambig_results, ambig_results, not_found_results
    sponsors_df = pd.read_csv(file_name, encoding=encoding)
    # columns to be saved in the output file
    res_columns = ['sponsor_name', 'city', 'county', 'found_name', 'status', 'company_number', 'reg_address',
                   'business_activity', 'sic_codes', 'sic_description', 'confidence']
    i, batch_num = 0, 0
    for index, row in sponsors_df[['name', 'city', 'county']].drop_duplicates().iterrows():
        search_company(row)
        i += 1
        if i >= BATCH_SIZE:
            append_unambig(res_columns, batch_num, f'results_good_{file_name[-10:-4]}.csv')
            i = 0
            batch_num += 1
            unambig_results = []
    append_unambig(res_columns, batch_num, f'results_good_{file_name[-10:-4]}.csv')

    print(f'{not_obv}  were not obvious / {not_found} were not found / {tot} total')
    print(f'\nAll found states list: {states}')

    with open(f'results_not_found_{file_name[-10:-4]}.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(not_found_results))
    with open(f'results_ambig_{file_name[-10:-4]}.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(ambig_results))


if __name__ == '__main__':
    file_name = 'tier-2-5_sponsors_200706.csv'
    # file_name = 'test.csv'
    get_sponsors_info(file_name)



