from bang_ai.boards import spawn_connected_random_walk_shapes


def test_import_package():
    import bang_ai

    assert bang_ai.__version__


def test_square_shape_generation():
    starts = [(0, 0), (1, 1), (2, 2)]

    def sample_start():
        return starts.pop(0) if starts else None

    def neighbors(tile):
        x, y = tile
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def valid(candidate, pending):
        return candidate not in pending

    shapes = spawn_connected_random_walk_shapes(
        shape_count=3,
        min_sections=2,
        max_sections=3,
        sample_start_fn=sample_start,
        neighbor_candidates_fn=neighbors,
        is_candidate_valid_fn=valid,
    )

    assert len(shapes) == 3
    assert all(len(shape) >= 2 for shape in shapes)
