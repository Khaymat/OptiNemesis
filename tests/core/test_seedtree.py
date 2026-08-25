import numpy as np
import pytest

from optinemesis.core import SeedTree


def test_subtree_labels_and_determinism() -> None:
    t1 = SeedTree.create(root_entropy=42)
    t2 = SeedTree.create(root_entropy=42)
    assert t1.problem_search.seeds(10) == t2.problem_search.seeds(10)
    assert t1.meta.seeds(5) == (42 and t2.meta.seeds(5))


def test_distinct_root_distinct_seeds() -> None:
    a = SeedTree.create(root_entropy=1).problem_search.seeds(8)
    b = SeedTree.create(root_entropy=2).problem_search.seeds(8)
    assert a != b


def test_search_validation_disjoint_by_construction() -> None:
    tree = SeedTree.create(root_entropy=7)
    n = 64
    search_problem = set(tree.problem_search.seeds(n))
    search_a = set(tree.opt_a.seeds(n))
    search_b = set(tree.opt_b.seeds(n))
    meta = set(tree.meta.seeds(n))
    val = tree.validation_subtrees()
    val_problem = set(val["problem_val"].seeds(n))
    val_a = set(val["opt_a_val"].seeds(n))
    val_b = set(val["opt_b_val"].seeds(n))

    groups = [search_problem, search_a, search_b, meta, val_problem, val_a, val_b]
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1 :]:
            assert not (g1 & g2), "seed subtrees must be pairwise disjoint"


def test_leaf_seeds_are_valid_rng_roots() -> None:
    tree = SeedTree.create(root_entropy=99)
    seeds = tree.problem_search.seeds(4)
    for s in seeds:
        rng = np.random.default_rng(s)
        x = rng.random(3)
        assert x.shape == (3,)


def test_generator_is_positional_deterministic() -> None:
    t1 = SeedTree.create(root_entropy=5)
    t2 = SeedTree.create(root_entropy=5)
    g1 = t1.opt_a.generator(3)
    g2 = t2.opt_a.generator(3)
    np.testing.assert_array_equal(g1.random(7), g2.random(7))


def test_create_without_entropy_is_usable() -> None:
    t = SeedTree.create()
    assert isinstance(int(t.root_entropy), int)
    assert len(t.problem_search.seeds(3)) == 3


def test_invalid_root_rejected() -> None:
    with pytest.raises(ValueError):
        SeedTree(root_entropy="not-a-number")


def test_recipe_string_present() -> None:
    from optinemesis.core import DERIVATION_RECIPE

    assert "validation" in DERIVATION_RECIPE


def test_to_dict() -> None:
    d = SeedTree.create(root_entropy=11).to_dict()
    assert d["root_entropy"] == "11"
