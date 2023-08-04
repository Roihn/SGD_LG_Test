MAX_TOKENS = 60 # The maximum number of tokens of utterances in the dataset

DATA_PATH = 'dialogues_001.json'

ACT_TO_ID = {
    'AFFIRM': 0,
    'OFFER_INTENT': 1,
    'INFORM_INTENT': 2,
    'REQUEST': 3,
    'NOTIFY_SUCCESS': 4,
    'REQUEST_ALTS': 5,
    'CONFIRM': 6,
    'GOODBYE': 7,
    'INFORM_COUNT': 8,
    'NOTIFY_FAILURE': 9,
    'AFFIRM_INTENT': 10,
    'REQ_MORE': 11,
    'OFFER': 12,
    'SELECT': 13,
    'THANK_YOU': 14,
    'INFORM': 15,
    'NEGATE': 16
}

SLOT_TO_ID = {
    'None': 0,
    'party_size': 1,
    'city': 2,
    'street_address': 3,
    'restaurant_name': 4,
    'phone_number': 5,
    'price_range': 6,
    'intent': 7,
    'serves_alcohol': 8,
    'date': 9,
    'count': 10,
    'cuisine': 11,
    'has_live_music': 12,
    'time': 13
}

ID_TO_ACT = {
    0: 'AFFIRM',
    1: 'OFFER_INTENT',
    2: 'INFORM_INTENT',
    3: 'REQUEST',
    4: 'NOTIFY_SUCCESS',
    5: 'REQUEST_ALTS',
    6: 'CONFIRM',
    7: 'GOODBYE',
    8: 'INFORM_COUNT',
    9: 'NOTIFY_FAILURE',
    10: 'AFFIRM_INTENT',
    11: 'REQ_MORE',
    12: 'OFFER',
    13: 'SELECT',
    14: 'THANK_YOU',
    15: 'INFORM',
    16: 'NEGATE'
}

ID_TO_SLOT = {
    0: 'None',
    1: 'party_size',
    2: 'city',
    3: 'street_address',
    4: 'restaurant_name',
    5: 'phone_number',
    6: 'price_range',
    7: 'intent',
    8: 'serves_alcohol',
    9: 'date',
    10: 'count',
    11: 'cuisine',
    12: 'has_live_music',
    13: 'time'
}

ACT_SLOT_PAIR_TO_ID = {
    'THANK_YOU/None': 0,
    'OFFER_INTENT/intent': 1,
    'INFORM/party_size': 2,
    'AFFIRM/None': 3,
    'INFORM_INTENT/intent': 4,
    'INFORM/street_address': 5,
    'REQ_MORE/None': 6,
    'INFORM/city': 7,
    'INFORM/has_live_music': 8,
    'OFFER/time': 9,
    'REQUEST/serves_alcohol': 10,
    'CONFIRM/date': 11,
    'OFFER/party_size': 12,
    'NOTIFY_SUCCESS/None': 13,
    'OFFER/city': 14,
    'CONFIRM/restaurant_name': 15,
    'INFORM/cuisine': 16,
    'SELECT/None': 17,
    'REQUEST/price_range': 18,
    'INFORM/phone_number': 19,
    'NOTIFY_FAILURE/None': 20,
    'REQUEST/time': 21,
    'CONFIRM/time': 22,
    'INFORM/date': 23,
    'REQUEST/street_address': 24,
    'NEGATE/None': 25,
    'INFORM/serves_alcohol': 26,
    'REQUEST/city': 27,
    'REQUEST/has_live_music': 28,
    'CONFIRM/party_size': 29,
    'INFORM_COUNT/count': 30,
    'CONFIRM/city': 31,
    'OFFER/date': 32,
    'REQUEST_ALTS/None': 33,
    'REQUEST/cuisine': 34,
    'GOODBYE/None': 35,
    'INFORM/price_range': 36,
    'OFFER/restaurant_name': 37,
    'REQUEST/phone_number': 38,
    'AFFIRM_INTENT/None': 39,
    'INFORM/time': 40
}

ID_TO_ACT_SLOT_PAIR = {
    0: 'THANK_YOU/None', 
    1: 'OFFER_INTENT/intent', 
    2: 'INFORM/party_size', 
    3: 'AFFIRM/None', 
    4: 'INFORM_INTENT/intent', 
    5: 'INFORM/street_address', 
    6: 'REQ_MORE/None', 
    7: 'INFORM/city', 
    8: 'INFORM/has_live_music', 
    9: 'OFFER/time', 
    10: 'REQUEST/serves_alcohol', 
    11: 'CONFIRM/date', 
    12: 'OFFER/party_size', 
    13: 'NOTIFY_SUCCESS/None', 
    14: 'OFFER/city', 
    15: 'CONFIRM/restaurant_name', 
    16: 'INFORM/cuisine', 
    17: 'SELECT/None', 
    18: 'REQUEST/price_range', 
    19: 'INFORM/phone_number', 
    20: 'NOTIFY_FAILURE/None', 
    21: 'REQUEST/time', 
    22: 'CONFIRM/time', 
    23: 'INFORM/date', 
    24: 'REQUEST/street_address', 
    25: 'NEGATE/None', 
    26: 'INFORM/serves_alcohol', 
    27: 'REQUEST/city', 
    28: 'REQUEST/has_live_music', 
    29: 'CONFIRM/party_size', 
    30: 'INFORM_COUNT/count', 
    31: 'CONFIRM/city', 
    32: 'OFFER/date', 
    33: 'REQUEST_ALTS/None', 
    34: 'REQUEST/cuisine', 
    35: 'GOODBYE/None', 
    36: 'INFORM/price_range', 
    37: 'OFFER/restaurant_name', 
    38: 'REQUEST/phone_number', 
    39: 'AFFIRM_INTENT/None', 
    40: 'INFORM/time'
}