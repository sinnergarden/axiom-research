import sys
import unittest
from types import SimpleNamespace

from axiom_research.stock_matrix_feature_producer import _owned_bytes


class OwnedBytesTests(unittest.TestCase):
    def test_shared_containers_cycles_and_scalar_identity_have_exact_flat_total(self):
        payload = 'shared coverage payload'
        leaf = [payload, 0.0, -0.0, False, None]
        frozen = frozenset((41, 1001))
        pair = (leaf, frozen)
        cycle = []
        root = {'left': leaf, 'right': leaf, 'cycle': cycle, 'pair': pair}
        cycle.append(root)
        # Independently enumerate this small graph rather than repeat traversal.
        objects = [root, *root.keys(), leaf, *leaf, cycle, pair, frozen, *frozen]
        unique = {id(item): item for item in objects}
        seen = set()
        self.assertEqual(_owned_bytes(root, seen), sum(map(sys.getsizeof, unique.values())))
        self.assertEqual(seen, set(unique))

    def test_existing_seen_subgraph_is_not_descended(self):
        subtree = {'unseen child': ['unseen leaf']}
        root = [subtree, subtree, 79]
        seen = {id(subtree)}
        self.assertEqual(_owned_bytes(root, seen), sys.getsizeof(root)+sys.getsizeof(79))
        self.assertEqual(seen, {id(subtree), id(root), id(79)})
        self.assertEqual(_owned_bytes(root, seen), 0)

    def test_document_payload_and_separately_reachable_dict_keep_conservative_charge(self):
        payload = 'canonical Core payload'
        document = SimpleNamespace(payload=payload)
        fields = document.__dict__
        root = [document, payload, fields, document]
        expected = (sys.getsizeof(root)+sys.getsizeof(document)+sys.getsizeof(payload)
                    +2*sys.getsizeof(fields)+sys.getsizeof(next(iter(fields))))
        self.assertEqual(_owned_bytes(root), expected)

    def test_fresh_measurement_observes_mutation_without_retaining_a_cache(self):
        leaf = ['original']
        root = {'metadata': leaf}
        before = _owned_bytes(root)
        added = {'new source': 'new evidence'}
        leaf.append(added)
        objects = [root, *root.keys(), leaf, *leaf, *added.keys(), *added.values()]
        unique = {id(item): item for item in objects}
        after = _owned_bytes(root)
        self.assertGreater(after, before)
        self.assertEqual(after, sum(map(sys.getsizeof, unique.values())))
