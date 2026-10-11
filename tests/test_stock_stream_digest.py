"""Streaming hashes preserve existing canonical bytes, including adapter defaults."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import unittest

from axiom_research.stock_artifacts import digest, digest_stream
from axiom_research.data_adapter import _digest


class StreamDigestTests(unittest.TestCase):
    def test_nested_unicode_negative_zero_and_scalar_bytes(self):
        value = {'records':[{'security_id':'上海😀', 'values':[-0.0,1e-20,None,True]}],
                 'field_meta':{'close':{'by_key':[{'usable_from':'2023-01-03T20:00:00+08:00'}]}},
                 'context':{'a':1, 'z':'\\n'}}
        self.assertEqual(digest_stream(value), digest(value))
        self.assertEqual(_digest(value), digest(value))
        value['clock'] = datetime(2023,1,3,tzinfo=timezone.utc)
        old = 'sha256:'+sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
            ensure_ascii=False,allow_nan=False,default=str).encode()).hexdigest()
        self.assertEqual(_digest(value), old)

    def test_nonfinite_still_rejected(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(ValueError): digest_stream({'value':value})


if __name__ == '__main__': unittest.main()
