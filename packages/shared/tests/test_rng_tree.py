import pytest
from esports_sim.rng import RngTree


def _draw(node, n=4):
    return node.generator().integers(0, 1_000_000_000, size=n).tolist()


def test_same_seed_same_path_same_output():
    a = RngTree(seed=42).at("matches/m001")
    b = RngTree(seed=42).at("matches/m001")
    assert _draw(a) == _draw(b)


def test_different_seed_different_output():
    a = RngTree(seed=42).at("matches/m001")
    b = RngTree(seed=43).at("matches/m001")
    assert _draw(a) != _draw(b)


def test_path_equivalence():
    root = RngTree(seed=42)
    chained = root.child("matches").child("m001").child("round1")
    walked = root.at("matches/m001/round1")
    assert _draw(chained) == _draw(walked)


def test_siblings_are_independent():
    root = RngTree(seed=42)
    assert _draw(root.child("a")) != _draw(root.child("b"))


def test_visiting_a_sibling_does_not_perturb_others():
    root1 = RngTree(seed=42)
    root2 = RngTree(seed=42)
    # root2 visits an unrelated sibling first; root1 does not.
    _draw(root2.child("unrelated"))
    assert _draw(root1.child("matches")) == _draw(root2.child("matches"))


def test_child_caching_returns_same_node():
    root = RngTree(seed=42)
    assert root.child("x") is root.child("x")


def test_generator_called_twice_starts_from_same_state():
    node = RngTree(seed=42).at("a")
    assert _draw(node) == _draw(node)


def test_invalid_label_rejected():
    root = RngTree(seed=42)
    with pytest.raises(ValueError):
        root.child("")
    with pytest.raises(ValueError):
        root.child("a/b")


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        RngTree(seed=-1)


def test_root_path_is_empty():
    assert RngTree(seed=1).path == ()
    assert RngTree(seed=1).at("a/b").path == ("a", "b")


def test_zero_seed_works():
    # Edge case: bit_length of 0 is 0; constructor must still produce 32 bytes.
    a = RngTree(seed=0).at("x")
    b = RngTree(seed=0).at("x")
    assert _draw(a) == _draw(b)
