---
applyTo: "tests/**/*.py"
---

# Test Code Review

## Test Organization
- Unit tests: no model download, no network, no disk I/O beyond tmp_path
- Integration tests: marked with `@pytest.mark.integration`, may download models
- Session-scoped fixtures for expensive model instantiation
- Each test class groups related tests (e.g., `TestDINOv2EmbedderIntegration`)

## Test Quality
- Every test has a Google-style docstring explaining what it verifies
- Use `torch.testing.assert_close` for tensor comparisons, not manual tolerance checks
- Use `np.testing.assert_array_equal` for numpy array comparisons
- Random test data fixtures should document they are unnormalized smoke tests
- No test fakes, mocks, or noop classes in production code (`src/`)

## Fixtures
- Fixture docstrings must accurately describe what the fixture provides
- Do not claim weight sharing between separately instantiated models
- `tmp_path` for all file I/O in tests
