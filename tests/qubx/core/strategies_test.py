import pandas as pd
from qubx import lookup
from qubx.core.basics import Deal, Instrument, Position
from qubx.impl.utils import ccxt_convert_deal_info, ccxt_convert_order_info, ccxt_restore_position_from_deals

C1 = {'info': {'e': 'executionReport', 'E': 1712231084293, 's': 'ACAUSDT', 'c': 'x-R4BD3S8238819a3237ee3b4e56266c', 'S': 'BUY', 'o': 'MARKET', 'f': 'GTC', 'q': '50.00000000', 'p': '0.00000000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': '', 'x': 'NEW', 'X': 'NEW', 'r': 'NONE', 'i': 149024288, 'l': '0.00000000', 'z': '0.00000000', 'L': '0.00000000', 'n': '0', 'N': None, 'T': 1712231084293, 't': -1, 'I': 315492989, 'w': True, 'm': False, 'M': False, 'O': 1712231084293, 'Z': '0.00000000', 'Y': '0.00000000', 'Q': '0.00000000', 'W': 1712231084293, 'V': 'EXPIRE_MAKER'}, 'symbol': 'ACA/USDT', 'id': '149024288', 'clientOrderId': 'x-R4BD3S8238819a3237ee3b4e56266c', 'timestamp': 1712231084293, 'datetime': '2024-04-04T11:44:44.293Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712231084293, 'type': 'market', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': None, 'stopPrice': 0.0, 'triggerPrice': 0.0, 'amount': 50.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 50.0, 'status': 'open', 'fee': None, 'trades': [], 'fees': [], 'takeProfitPrice': None, 'stopLossPrice': None}

C2 = {'info': {'e': 'executionReport', 'E': 1712231084293, 's': 'ACAUSDT', 'c': 'x-R4BD3S8238819a3237ee3b4e56266c', 'S': 'BUY', 'o': 'MARKET', 'f': 'GTC', 'q': '50.00000000', 'p': '0.00000000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': '', 'x': 'TRADE', 'X': 'FILLED', 'r': 'NONE', 'i': 149024288, 'l': '50.00000000', 'z': '50.00000000', 'L': '0.16120000', 'n': '0.00001027', 'N': 'BNB', 'T': 1712231084293, 't': 17285573, 'I': 315492990, 'w': False, 'm': False, 'M': True, 'O': 1712231084293, 'Z': '8.06000000', 'Y': '8.06000000', 'Q': '0.00000000', 'W': 1712231084293, 'V': 'EXPIRE_MAKER'}, 'symbol': 'ACA/USDT', 'id': '149024288', 'clientOrderId': 'x-R4BD3S8238819a3237ee3b4e56266c', 'timestamp': 1712231084293, 'datetime': '2024-04-04T11:44:44.293Z', 'lastTradeTimestamp': 1712231084293, 'lastUpdateTimestamp': 1712231084293, 'type': 'market', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.1612, 'stopPrice': 0.0, 'triggerPrice': 0.0, 'amount': 50.0, 'cost': 8.06, 'average': 0.1612, 'filled': 50.0, 'remaining': 0.0, 'status': 'closed', 'fee': {'cost': 1.027e-05, 'currency': 'BNB'}, 'trades': [{'info': {'e': 'executionReport', 'E': 1712231084293, 's': 'ACAUSDT', 'c': 'x-R4BD3S8238819a3237ee3b4e56266c', 'S': 'BUY', 'o': 'MARKET', 'f': 'GTC', 'q': '50.00000000', 'p': '0.00000000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': '', 'x': 'TRADE', 'X': 'FILLED', 'r': 'NONE', 'i': 149024288, 'l': '50.00000000', 'z': '50.00000000', 'L': '0.16120000', 'n': '0.00001027', 'N': 'BNB', 'T': 1712231084293, 't': 17285573, 'I': 315492990, 'w': False, 'm': False, 'M': True, 'O': 1712231084293, 'Z': '8.06000000', 'Y': '8.06000000', 'Q': '0.00000000', 'W': 1712231084293, 'V': 'EXPIRE_MAKER'}, 'timestamp': 1712231084293, 'datetime': '2024-04-04T11:44:44.293Z', 'symbol': 'ACA/USDT', 'id': '17285573', 'order': '149024288', 'type': 'market', 'takerOrMaker': 'taker', 'side': 'buy', 'price': 0.1612, 'amount': 50.0, 'cost': 8.06, 'fee': {'cost': 1.027e-05, 'currency': 'BNB'}, 'fees': [{'cost': 1.027e-05, 'currency': 'BNB'}]}], 'fees': [], 'takeProfitPrice': None, 'stopLossPrice': None}

C3={'info': {'e': 'executionReport', 'E': 1712231324395, 's': 'ACAUSDT', 'c': 'x-ZQ3RVWN66045debc8d86be46f76afa', 'S': 'SELL', 'o': 'MARKET', 'f': 'GTC', 'q': '50.00000000', 'p': '0.00000000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': '', 'x': 'NEW', 'X': 'NEW', 'r': 'NONE', 'i': 149024642, 'l': '0.00000000', 'z': '0.00000000', 'L': '0.00000000', 'n': '0', 'N': None, 'T': 1712231324394, 't': -1, 'I': 315493717, 'w': True, 'm': False, 'M': False, 'O': 1712231324394, 'Z': '0.00000000', 'Y': '0.00000000', 'Q': '0.00000000', 'W': 1712231324394, 'V': 'EXPIRE_MAKER'}, 'symbol': 'ACA/USDT', 'id': '149024642', 'clientOrderId': 'x-ZQ3RVWN66045debc8d86be46f76afa', 'timestamp': 1712231324394, 'datetime': '2024-04-04T11:48:44.394Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712231324394, 'type': 'market', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'sell', 'price': None, 'stopPrice': 0.0, 'triggerPrice': 0.0, 'amount': 50.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 50.0, 'status': 'open', 'fee': None, 'trades': [], 'fees': [], 'takeProfitPrice': None, 'stopLossPrice': None}

C4={'info': {'e': 'executionReport', 'E': 1712231324395, 's': 'ACAUSDT', 'c': 'x-ZQ3RVWN66045debc8d86be46f76afa', 'S': 'SELL', 'o': 'MARKET', 'f': 'GTC', 'q': '50.00000000', 'p': '0.00000000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': '', 'x': 'TRADE', 'X': 'FILLED', 'r': 'NONE', 'i': 149024642, 'l': '50.00000000', 'z': '50.00000000', 'L': '0.16060000', 'n': '0.00001026', 'N': 'BNB', 'T': 1712231324394, 't': 17285596, 'I': 315493718, 'w': False, 'm': False, 'M': True, 'O': 1712231324394, 'Z': '8.03000000', 'Y': '8.03000000', 'Q': '0.00000000', 'W': 1712231324394, 'V': 'EXPIRE_MAKER'}, 'symbol': 'ACA/USDT', 'id': '149024642', 'clientOrderId': 'x-ZQ3RVWN66045debc8d86be46f76afa', 'timestamp': 1712231324394, 'datetime': '2024-04-04T11:48:44.394Z', 'lastTradeTimestamp': 1712231324394, 'lastUpdateTimestamp': 1712231324394, 'type': 'market', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'sell', 'price': 0.1606, 'stopPrice': 0.0, 'triggerPrice': 0.0, 'amount': 50.0, 'cost': 8.03, 'average': 0.1606, 'filled': 50.0, 'remaining': 0.0, 'status': 'closed', 'fee': {'cost': 1.026e-05, 'currency': 'BNB'}, 'trades': [{'info': {'e': 'executionReport', 'E': 1712231324395, 's': 'ACAUSDT', 'c': 'x-ZQ3RVWN66045debc8d86be46f76afa', 'S': 'SELL', 'o': 'MARKET', 'f': 'GTC', 'q': '50.00000000', 'p': '0.00000000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': '', 'x': 'TRADE', 'X': 'FILLED', 'r': 'NONE', 'i': 149024642, 'l': '50.00000000', 'z': '50.00000000', 'L': '0.16060000', 'n': '0.00001026', 'N': 'BNB', 'T': 1712231324394, 't': 17285596, 'I': 315493718, 'w': False, 'm': False, 'M': True, 'O': 1712231324394, 'Z': '8.03000000', 'Y': '8.03000000', 'Q': '0.00000000', 'W': 1712231324394, 'V': 'EXPIRE_MAKER'}, 'timestamp': 1712231324394, 'datetime': '2024-04-04T11:48:44.394Z', 'symbol': 'ACA/USDT', 'id': '17285596', 'order': '149024642', 'type': 'market', 'takerOrMaker': 'taker', 'side': 'sell', 'price': 0.1606, 'amount': 50.0, 'cost': 8.03, 'fee': {'cost': 1.026e-05, 'currency': 'BNB'}, 'fees': [{'cost': 1.026e-05, 'currency': 'BNB'}]}], 'fees': [], 'takeProfitPrice': None, 'stopLossPrice': None}

C5new={
    'info': {
        'e': 'executionReport', 'E': 1712231523596, 's': 'ACAUSDT', 'c': 'x-R4BD3S82e78e232b573fe3c2a48153', 'S': 'BUY', 'o': 'LIMIT', 'f': 'GTC', 'q': '51.00000000', 'p': '0.16110000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': '', 'x': 'NEW', 'X': 'NEW', 'r': 'NONE', 'i': 149025236, 'l': '0.00000000', 'z': '0.00000000', 'L': '0.00000000', 'n': '0', 'N': None, 'T': 1712231523596, 't': -1, 'I': 315494940, 'w': True, 'm': False, 'M': False, 'O': 1712231523596, 'Z': '0.00000000', 'Y': '0.00000000', 'Q': '0.00000000', 'W': 1712231523596, 'V': 'EXPIRE_MAKER'
    }, 
    'symbol': 'ACA/USDT', 
    'id': '149025236', 'clientOrderId': 'x-R4BD3S82e78e232b573fe3c2a48153', 
    'timestamp': 1712231523596, 'datetime': '2024-04-04T11:52:03.596Z', 
    'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712231523596, 
    'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.1611, 'stopPrice': 0.0, 
    'triggerPrice': 0.0, 'amount': 51.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 51.0, 
    'status': 'open', 'fee': None, 'trades': [], 'fees': [], 'takeProfitPrice': None, 'stopLossPrice': None
    }

C6ex={
    'info': {
        'e': 'executionReport', 'E': 1712231530098, 's': 'ACAUSDT', 'c': 'x-R4BD3S82e78e232b573fe3c2a48153', 'S': 'BUY', 'o': 'LIMIT', 'f': 'GTC', 'q': '51.00000000', 'p': '0.16110000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': '', 'x': 'TRADE', 'X': 'FILLED', 'r': 'NONE', 'i': 149025236, 'l': '51.00000000', 'z': '51.00000000', 'L': '0.16110000', 'n': '0.00001044', 'N': 'BNB', 'T': 1712231530082, 't': 17285628, 'I': 315494958, 'w': False, 'm': True, 'M': True, 'O': 1712231523596, 'Z': '8.21610000', 'Y': '8.21610000', 'Q': '0.00000000', 'W': 1712231523596, 'V': 'EXPIRE_MAKER'
    }, 
    'symbol': 'ACA/USDT', 
    'id': '149025236', 'clientOrderId': 'x-R4BD3S82e78e232b573fe3c2a48153', 
    'timestamp': 1712231523596, 
    'datetime': '2024-04-04T11:52:03.596Z', 
    'lastTradeTimestamp': 1712231530082, 
    'lastUpdateTimestamp': 1712231530082, 
    'type': 'limit', 
    'timeInForce': 'GTC', 
    'postOnly': False, 
    'reduceOnly': None, 
    'side': 'buy', 
    'price': 0.1611, 
    'stopPrice': 0.0, 
    'triggerPrice': 0.0, 
    'amount': 51.0, 
    'cost': 8.2161, 
    'average': 0.1611, 
    'filled': 51.0, 
    'remaining': 0.0, 
    'status': 'closed', 
    'fee': {'cost': 1.044e-05, 'currency': 'BNB'}, 
    'trades': [
        {
            'info': {
                'e': 'executionReport', 'E': 1712231530098, 's': 'ACAUSDT', 'c': 'x-R4BD3S82e78e232b573fe3c2a48153', 'S': 'BUY', 'o': 'LIMIT', 'f': 'GTC', 'q': '51.00000000', 'p': '0.16110000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': '', 'x': 'TRADE', 'X': 'FILLED', 'r': 'NONE', 'i': 149025236, 'l': '51.00000000', 'z': '51.00000000', 'L': '0.16110000', 'n': '0.00001044', 'N': 'BNB', 'T': 1712231530082, 't': 17285628, 'I': 315494958, 'w': False, 'm': True, 'M': True, 'O': 1712231523596, 'Z': '8.21610000', 'Y': '8.21610000', 'Q': '0.00000000', 'W': 1712231523596, 'V': 'EXPIRE_MAKER'
            }, 
            'timestamp': 1712231530082, 'datetime': '2024-04-04T11:52:10.082Z', 
            'symbol': 'ACA/USDT', 
            'id': '17285628', 'order': '149025236', 
            'type': 'limit', 
            'takerOrMaker': 'maker', 
            'side': 'buy', 
            'price': 0.1611, 
            'amount': 51.0, 'cost': 8.2161, 
            'fee': {'cost': 1.044e-05, 'currency': 'BNB'}, 'fees': [{'cost': 1.044e-05, 'currency': 'BNB'}]
        }
    ], 
    'fees': [], 'takeProfitPrice': None, 'stopLossPrice': None}

C7cancel= {
    'info': {'e': 'executionReport', 'E': 1712236270257, 's': 'ACAUSDT', 'c': 'JzCqzi9q0LSiyWH8HJuYtQ', 'S': 'BUY', 'o': 'LIMIT', 'f': 'GTC', 'q': '50.00000000', 'p': '0.10000000', 'P': '0.00000000', 'F': '0.00000000', 'g': -1, 'C': 'x-R4BD3S82d0c6eae956a555dbab1b26', 'x': 'CANCELED', 'X': 'CANCELED', 'r': 'NONE', 'i': 149034846, 'l': '0.00000000', 'z': '0.00000000', 'L': '0.00000000', 'n': '0', 'N': None, 'T': 1712236270256, 't': -1, 'I': 315514865, 'w': False, 'm': False, 'M': False, 'O': 1712236254516, 'Z': '0.00000000', 'Y': '0.00000000', 'Q': '0.00000000', 'W': 1712236254516, 'V': 'EXPIRE_MAKER'}, 
    'symbol': 'ACA/USDT', 
    'id': '149034846', 'clientOrderId': 'x-R4BD3S82d0c6eae956a555dbab1b26',
    'timestamp': 1712236254516, 'datetime': '2024-04-04T13:10:54.516Z', 
    'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712236270256, 
    'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.1, 'stopPrice': 0.0, 
    'triggerPrice': 0.0, 'amount': 50.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 50.0, 'status': 'canceled', 
    'fee': None, 'trades': [], 'fees': [], 'takeProfitPrice': None, 'stopLossPrice': None
}

HIST = [
    {
        'info': {
            'symbol': 'ACAUSDT', 
            'orderListId': '-1', 
            'orderId': '148022194', 'clientOrderId': 'electron_ce63fa14b5da4f5a99ac60679e9', 
            'price': '0.00000000', 
            'origQty': '50.00000000', 
            'executedQty': '50.00000000', 
            'cummulativeQuoteQty': '8.81359600', 
            'status': 'FILLED', 
            'timeInForce': 'GTC', 
            'type': 'MARKET', 
            'side': 'BUY', 
            'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 
            'time': '1711978683891', 'updateTime': '1711978683891', 'isWorking': True, 'workingTime': '1711978683891', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'
            }, 
        'id': '148022194', 'clientOrderId': 'electron_ce63fa14b5da4f5a99ac60679e9', 
        'timestamp': 1711978683891, 
        'datetime': '2024-04-01T13:38:03.891Z', 
        'lastTradeTimestamp': 1711978683891, 
        'lastUpdateTimestamp': 1711978683891, 
        'symbol': 'ACA/USDT', 
        'type': 'market', 
        'timeInForce': 'GTC', 
        'postOnly': False, 'reduceOnly': None, 
        'side': 'buy', 
        'price': 0.17627192, 
        'amount': 50.0, 'cost': 8.813596, 'average': 0.17627192, 
        'filled': 50.0, 
        'remaining': 0.0, 
        'status': 'closed', 
        'fee': None, 
        'triggerPrice': None, 
        'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None
    }, 
    {'info': {'symbol': 'ACAUSDT', 'orderId': '148035855', 'orderListId': '-1', 'clientOrderId': 'electron_d29b06e17def4150ab57ee416e9', 'price': '0.17000000', 'origQty': '50.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1711982582392', 'updateTime': '1711983183356', 'isWorking': True, 'workingTime': '1711982582392', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '148035855', 'clientOrderId': 'electron_d29b06e17def4150ab57ee416e9', 'timestamp': 1711982582392, 'datetime': '2024-04-01T14:43:02.392Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1711983183356, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.17, 'triggerPrice': None, 'amount': 50.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 50.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '148039443', 'orderListId': '-1', 'clientOrderId': 'electron_834e5611d24246e281eec59b15b', 'price': '0.17160000', 'origQty': '50.00000000', 'executedQty': '50.00000000', 'cummulativeQuoteQty': '8.58000000', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1711983183356', 'updateTime': '1711983194502', 'isWorking': True, 'workingTime': '1711983183356', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '148039443', 'clientOrderId': 'electron_834e5611d24246e281eec59b15b', 'timestamp': 1711983183356, 'datetime': '2024-04-01T14:53:03.356Z', 'lastTradeTimestamp': 1711983194502, 'lastUpdateTimestamp': 1711983194502, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.1716, 'triggerPrice': None, 'amount': 50.0, 'cost': 8.58, 'average': 0.1716, 'filled': 50.0, 'remaining': 0.0, 'status': 'closed', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '148040032', 'orderListId': '-1', 'clientOrderId': 'electron_ef7cd323345e4251860252c0f9c', 'price': '0.16500000', 'origQty': '50.00000000', 'executedQty': '50.00000000', 'cummulativeQuoteQty': '8.25000000', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1711983237405', 'updateTime': '1712024700808', 'isWorking': True, 'workingTime': '1711983237405', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '148040032', 'clientOrderId': 'electron_ef7cd323345e4251860252c0f9c', 'timestamp': 1711983237405, 'datetime': '2024-04-01T14:53:57.405Z', 'lastTradeTimestamp': 1712024700808, 'lastUpdateTimestamp': 1712024700808, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.165, 'triggerPrice': None, 'amount': 50.0, 'cost': 8.25, 'average': 0.165, 'filled': 50.0, 'remaining': 0.0, 'status': 'closed', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '148365016', 'orderListId': '-1', 'clientOrderId': 'electron_be5d1a8530a64554bb39a7cd659', 'price': '0.15350000', 'origQty': '40.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'SELL', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712053738681', 'updateTime': '1712053742703', 'isWorking': True, 'workingTime': '1712053738681', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '148365016', 'clientOrderId': 'electron_be5d1a8530a64554bb39a7cd659', 'timestamp': 1712053738681, 'datetime': '2024-04-02T10:28:58.681Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712053742703, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'sell', 'price': 0.1535, 'triggerPrice': None, 'amount': 40.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 40.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '148365070', 'orderListId': '-1', 'clientOrderId': 'electron_433cdb795a03416ebf470ffe0c4', 'price': '0.00000000', 'origQty': '40.00000000', 'executedQty': '40.00000000', 'cummulativeQuoteQty': '6.13600000', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'SELL', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712053753883', 'updateTime': '1712053753883', 'isWorking': True, 'workingTime': '1712053753883', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '148365070', 'clientOrderId': 'electron_433cdb795a03416ebf470ffe0c4', 'timestamp': 1712053753883, 'datetime': '2024-04-02T10:29:13.883Z', 'lastTradeTimestamp': 1712053753883, 'lastUpdateTimestamp': 1712053753883, 'symbol': 'ACA/USDT', 'type': 'market', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'sell', 'price': 0.1534, 'triggerPrice': None, 'amount': 40.0, 'cost': 6.136, 'average': 0.1534, 'filled': 40.0, 'remaining': 0.0, 'status': 'closed', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '148365474', 'orderListId': '-1', 'clientOrderId': 'electron_97ca6fb0a05c46a39d334dbb5f9', 'price': '0.14000000', 'origQty': '50.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'NEW', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712053802532', 'updateTime': '1712053802532', 'isWorking': True, 'workingTime': '1712053802532', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '148365474', 'clientOrderId': 'electron_97ca6fb0a05c46a39d334dbb5f9', 'timestamp': 1712053802532, 'datetime': '2024-04-02T10:30:02.532Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712053802532, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.14, 'triggerPrice': None, 'amount': 50.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 50.0, 'status': 'open', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149002647', 'orderListId': '-1', 'clientOrderId': 'electron_e5cd4c7d8e6545329f6fcf4db7f', 'price': '0.11370000', 'origQty': '60.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712217613867', 'updateTime': '1712218053620', 'isWorking': True, 'workingTime': '1712217613867', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149002647', 'clientOrderId': 'electron_e5cd4c7d8e6545329f6fcf4db7f', 'timestamp': 1712217613867, 'datetime': '2024-04-04T08:00:13.867Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712218053620, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.1137, 'triggerPrice': None, 'amount': 60.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 60.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149003297', 'orderListId': '-1', 'clientOrderId': 'electron_cb3dbee6e7bc4b929ff44ba3786', 'price': '0.11000000', 'origQty': '50.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712217986030', 'updateTime': '1712218300555', 'isWorking': True, 'workingTime': '1712217986030', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149003297', 'clientOrderId': 'electron_cb3dbee6e7bc4b929ff44ba3786', 'timestamp': 1712217986030, 'datetime': '2024-04-04T08:06:26.030Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712218300555, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 50.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 50.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149003508', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S825a12fabde78459ddba663d', 'price': '0.11000000', 'origQty': '50.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712218303965', 'updateTime': '1712218309978', 'isWorking': True, 'workingTime': '1712218303965', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149003508', 'clientOrderId': 'x-R4BD3S825a12fabde78459ddba663d', 'timestamp': 1712218303965, 'datetime': '2024-04-04T08:11:43.965Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712218309978, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 50.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 50.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149003517', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S826e4d8ad867cb3c44bce1a4', 'price': '0.11000000', 'origQty': '55.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712218318244', 'updateTime': '1712218322103', 'isWorking': True, 'workingTime': '1712218318244', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149003517', 'clientOrderId': 'x-R4BD3S826e4d8ad867cb3c44bce1a4', 'timestamp': 1712218318244, 'datetime': '2024-04-04T08:11:58.244Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712218322103, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 55.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 55.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149005615', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S8291212ea87de91a9f9f35b9', 'price': '0.11000000', 'origQty': '55.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712219801356', 'updateTime': '1712219814771', 'isWorking': True, 'workingTime': '1712219801356', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149005615', 'clientOrderId': 'x-R4BD3S8291212ea87de91a9f9f35b9', 'timestamp': 1712219801356, 'datetime': '2024-04-04T08:36:41.356Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712219814771, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 55.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 55.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149005940', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S82d358df06012029342af699', 'price': '0.11000000', 'origQty': '56.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712219987403', 'updateTime': '1712220007662', 'isWorking': True, 'workingTime': '1712219987403', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149005940', 'clientOrderId': 'x-R4BD3S82d358df06012029342af699', 'timestamp': 1712219987403, 'datetime': '2024-04-04T08:39:47.403Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712220007662, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 56.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 56.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149005942', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S8269271c19ae84e81cb23ac2', 'price': '0.11000000', 'origQty': '56.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712220000623', 'updateTime': '1712220005289', 'isWorking': True, 'workingTime': '1712220000623', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149005942', 'clientOrderId': 'x-R4BD3S8269271c19ae84e81cb23ac2', 'timestamp': 1712220000623, 'datetime': '2024-04-04T08:40:00.623Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712220005289, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 56.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 56.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149006079', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S82d0e7c36f25751c5aabf7da', 'price': '0.11000000', 'origQty': '56.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712220151865', 'updateTime': '1712220841484', 'isWorking': True, 'workingTime': '1712220151865', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149006079', 'clientOrderId': 'x-R4BD3S82d0e7c36f25751c5aabf7da', 'timestamp': 1712220151865, 'datetime': '2024-04-04T08:42:31.865Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712220841484, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 56.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 56.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149006108', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S821d469253294faca6c229ea', 'price': '0.11000000', 'origQty': '56.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712220197484', 'updateTime': '1712220838930', 'isWorking': True, 'workingTime': '1712220197484', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149006108', 'clientOrderId': 'x-R4BD3S821d469253294faca6c229ea', 'timestamp': 1712220197484, 'datetime': '2024-04-04T08:43:17.484Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712220838930, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 56.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 56.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149006926', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S82bbbb6b31ab5998a19cfae6', 'price': '0.11000000', 'origQty': '56.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712220633889', 'updateTime': '1712220836233', 'isWorking': True, 'workingTime': '1712220633889', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149006926', 'clientOrderId': 'x-R4BD3S82bbbb6b31ab5998a19cfae6', 'timestamp': 1712220633889, 'datetime': '2024-04-04T08:50:33.889Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712220836233, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 56.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 56.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149009935', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S82879681022571d08019d38a', 'price': '0.11000000', 'origQty': '55.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712222504340', 'updateTime': '1712222709021', 'isWorking': True, 'workingTime': '1712222504340', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149009935', 'clientOrderId': 'x-R4BD3S82879681022571d08019d38a', 'timestamp': 1712222504340, 'datetime': '2024-04-04T09:21:44.340Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712222709021, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 55.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 55.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149010539', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S828ddb6978db78991bff1a9e', 'price': '0.11000000', 'origQty': '51.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712222720987', 'updateTime': '1712222738934', 'isWorking': True, 'workingTime': '1712222720987', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149010539', 'clientOrderId': 'x-R4BD3S828ddb6978db78991bff1a9e', 'timestamp': 1712222720987, 'datetime': '2024-04-04T09:25:20.987Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712222738934, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 51.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 51.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149011002', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S8216e2470fc1ce89641bf5f4', 'price': '0.11000000', 'origQty': '51.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712223054010', 'updateTime': '1712223290267', 'isWorking': True, 'workingTime': '1712223054010', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149011002', 'clientOrderId': 'x-R4BD3S8216e2470fc1ce89641bf5f4', 'timestamp': 1712223054010, 'datetime': '2024-04-04T09:30:54.010Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712223290267, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 51.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 51.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149012414', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S82bcfdd5c8e2fc51b0f1153', 'price': '0.11000000', 'origQty': '51.00000000', 'executedQty': '0.00000000', 'cummulativeQuoteQty': '0.00000000', 'status': 'CANCELED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712223436224', 'updateTime': '1712223697442', 'isWorking': True, 'workingTime': '1712223436224', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149012414', 'clientOrderId': 'x-R4BD3S82bcfdd5c8e2fc51b0f1153', 'timestamp': 1712223436224, 'datetime': '2024-04-04T09:37:16.224Z', 'lastTradeTimestamp': None, 'lastUpdateTimestamp': 1712223697442, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.11, 'triggerPrice': None, 'amount': 51.0, 'cost': 0.0, 'average': None, 'filled': 0.0, 'remaining': 51.0, 'status': 'canceled', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149024288', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S8238819a3237ee3b4e56266c', 'price': '0.00000000', 'origQty': '50.00000000', 'executedQty': '50.00000000', 'cummulativeQuoteQty': '8.06000000', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712231084293', 'updateTime': '1712231084293', 'isWorking': True, 'workingTime': '1712231084293', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149024288', 'clientOrderId': 'x-R4BD3S8238819a3237ee3b4e56266c', 'timestamp': 1712231084293, 'datetime': '2024-04-04T11:44:44.293Z', 'lastTradeTimestamp': 1712231084293, 'lastUpdateTimestamp': 1712231084293, 'symbol': 'ACA/USDT', 'type': 'market', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.1612, 'triggerPrice': None, 'amount': 50.0, 'cost': 8.06, 'average': 0.1612, 'filled': 50.0, 'remaining': 0.0, 'status': 'closed', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149024642', 'orderListId': '-1', 'clientOrderId': 'x-ZQ3RVWN66045debc8d86be46f76afa', 'price': '0.00000000', 'origQty': '50.00000000', 'executedQty': '50.00000000', 'cummulativeQuoteQty': '8.03000000', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'SELL', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712231324394', 'updateTime': '1712231324394', 'isWorking': True, 'workingTime': '1712231324394', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149024642', 'clientOrderId': 'x-ZQ3RVWN66045debc8d86be46f76afa', 'timestamp': 1712231324394, 'datetime': '2024-04-04T11:48:44.394Z', 'lastTradeTimestamp': 1712231324394, 'lastUpdateTimestamp': 1712231324394, 'symbol': 'ACA/USDT', 'type': 'market', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'sell', 'price': 0.1606, 'triggerPrice': None, 'amount': 50.0, 'cost': 8.03, 'average': 0.1606, 'filled': 50.0, 'remaining': 0.0, 'status': 'closed', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
    {'info': {'symbol': 'ACAUSDT', 'orderId': '149025236', 'orderListId': '-1', 'clientOrderId': 'x-R4BD3S82e78e232b573fe3c2a48153', 'price': '0.16110000', 'origQty': '51.00000000', 'executedQty': '51.00000000', 'cummulativeQuoteQty': '8.21610000', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'LIMIT', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': '1712231523596', 'updateTime': '1712231530082', 'isWorking': True, 'workingTime': '1712231523596', 'origQuoteOrderQty': '0.00000000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}, 'id': '149025236', 'clientOrderId': 'x-R4BD3S82e78e232b573fe3c2a48153', 'timestamp': 1712231523596, 'datetime': '2024-04-04T11:52:03.596Z', 'lastTradeTimestamp': 1712231530082, 'lastUpdateTimestamp': 1712231530082, 'symbol': 'ACA/USDT', 'type': 'limit', 'timeInForce': 'GTC', 'postOnly': False, 'reduceOnly': None, 'side': 'buy', 'price': 0.1611, 'triggerPrice': None, 'amount': 51.0, 'cost': 8.2161, 'average': 0.1611, 'filled': 51.0, 'remaining': 0.0, 'status': 'closed', 'fee': None, 'trades': [], 'fees': [], 'stopPrice': None, 'takeProfitPrice': None, 'stopLossPrice': None},
]


class TestStrats:
    
    def test_ccxt_exec_report_conversion(self):
        # - execution reports
        for o in [
            ccxt_convert_order_info('ACAUSDT', C1), 
            ccxt_convert_order_info('ACAUSDT', C2),
            ccxt_convert_order_info('ACAUSDT', C3),
            ccxt_convert_order_info('ACAUSDT', C4),
            ccxt_convert_order_info('ACAUSDT', C5new),
            ccxt_convert_order_info('ACAUSDT', C6ex),
            ccxt_convert_order_info('ACAUSDT', C7cancel),
        ]:
            print(o)
        print('-' * 50)

        print(ccxt_convert_order_info('ACAUSDT', C5new))
        print(ccxt_convert_order_info('ACAUSDT', C6ex))
        print(ccxt_convert_order_info('ACAUSDT', C7cancel))

        print('#' * 50)

        # - historical records
        pos = 0
        for h in HIST:
            o = ccxt_convert_order_info(h['info']['symbol'], h)
            if o.execution:
                print(o.execution)
            if o.status.upper() == 'CLOSED':
                print(o.quantity, o.execution.amount)
    
    def test_ccxt_hist_trades_conversion(self):
        raw = {
            'info': {
                'symbol': 'RAYUSDT', 'id': '56324015', 'orderId': '536752004', 'orderListId': '-1', 'price': '2.11290000', 'qty': '2.40000000', 'quoteQty': '5.07096000', 'commission': '0.00000648', 'commissionAsset': 'BNB', 'time': '1712497717270', 'isBuyer': True, 'isMaker': False, 'isBestMatch': True}, 
            'timestamp': 1712497717270, 
            'datetime': '2024-04-07T13:48:37.270Z', 
            'symbol': 'RAY/USDT', 
            'id': '56324015', 
            'order': '536752004', 
            'type': None, 
            'side': 'buy', 
            'takerOrMaker': 'taker', 
            'price': 2.1129, 
            'amount': 2.4, 
            'cost': 5.07096, 
            'fee': {'cost': 6.48e-06, 'currency': 'BNB'}, 
            'fees': [
                {'cost': 6.48e-06, 'currency': 'BNB'}
            ]
        }
        print(ccxt_convert_deal_info(raw))

    def test_position_restoring_from_deals(self):
        deals = [
            Deal(time=pd.Timestamp('2024-04-07 13:04:36.975000'), amount=0.5, price=180.84, aggressive=True, fee_amount=0.00011542, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 13:09:22.644000'), amount=-0.5, price=181.12, aggressive=True, fee_amount=0.00011562, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 13:48:37.611000'), amount=0.11, price=181.67, aggressive=True, fee_amount=2.544e-05, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 13:48:37.611000'), amount=0.11, price=181.68, aggressive=True, fee_amount=2.544e-05, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 13:48:37.611000'), amount=0.11, price=181.69, aggressive=True, fee_amount=2.544e-05, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 13:48:37.611000'), amount=0.22, price=181.69, aggressive=True, fee_amount=5.09e-05, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 14:12:34.624000'), amount=-0.55, price=181.29, aggressive=True, fee_amount=0.00012728, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 14:16:46.048000'), amount=0.7, price=181.32, aggressive=True, fee_amount=0.00016175, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 14:17:47.396000'), amount=-0.7, price=181.36, aggressive=True, fee_amount=0.00016176, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 14:18:25.864000'), amount=0.13, price=181.36, aggressive=True, fee_amount=3.005e-05, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 14:18:25.864000'), amount=0.11, price=181.36, aggressive=True, fee_amount=2.543e-05, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 14:18:25.864000'), amount=0.76, price=181.36, aggressive=True, fee_amount=0.00076, fee_currency='SOL'), # type: ignore
        ]

        instr1: Instrument = lookup.find_symbol('BINANCE', 'SOLUSDT') # type: ignore
        pos1 = Position(instr1, lookup.find_fees('binance', 'vip0_bnb')) # type: ignore
        vol1 = 0.99924

        pos1 = ccxt_restore_position_from_deals(pos1, vol1, deals)
        assert pos1.quantity == vol1

        deals = [
            Deal(time=pd.Timestamp('2024-04-07 12:40:41.717000'), amount=0.154, price=587.1, aggressive=True, fee_amount=0.0001155, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 12:41:59.307000'), amount=-0.153, price=586.6, aggressive=True, fee_amount=0.00011472, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-07 13:44:45.991000'), amount=-0.199, price=588.5, aggressive=True, fee_amount=0.00014922, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-08 12:45:49.738000'), amount=0.025, price=594.1, aggressive=True, fee_amount=1.875e-05, fee_currency='BNB'), # type: ignore
            Deal(time=pd.Timestamp('2024-04-08 12:48:37.543000'), amount=0.011, price=594.0, aggressive=True, fee_amount=8.25e-06, fee_currency='BNB'), # type: ignore
        ]

        instr2 = lookup.find_symbol('BINANCE', 'BNBUSDT') # type: ignore
        pos2 = Position(instr2, lookup.find_fees('binance', 'vip0_bnb')) # type: ignore
        vol2 = 0.035973

        pos2 = ccxt_restore_position_from_deals(pos2, vol2, deals);
        assert pos2.quantity == vol2




