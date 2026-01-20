# Linting Results Summary

## Python Linters (flake8, pylint, mypy)

### flake8
- **Files with issues**: `precompute_embeddings.py`, `run.py`, `server.py`, `src/cli_menu.py`, `src/clustering/clustering.py`, `src/clustering/dbscan_clustering.py`, `src/clustering/hdbscan_clustering.py`, `src/clustering/parallel_hdbscan_clustering.py`, `src/database/manager.py`, `src/processing/TFIDF.py`, `src/processing/dataset_filtering.py`, `src/processing/embedding_service.py`, `src/processing/llm_labelling.py`, `src/processing/remove_nonsignificative_words.py`, `src/processing/time_filtering.py`, `src/utils/hull_logic.py`, `src/visualization/temporal_analysis.py`, `tests/test_remove_nonsignificative_words.py`, `tests/test_server_features.py`, `tests/test_time_filtering.py`
- **Common issues**: 
  - Line too long (E501)
  - Trailing whitespace (W291)
  - Missing 2 blank lines (E302, E305)
  - Whitespace in blank lines (W293)
  - Inline comment spacing (E261)
  - Unused variables (F841)

### pylint
- **Files with issues**: `src/processing/embedding_service.py` (syntax error - possible false positive), `tests/test_remove_nonsignificative_words.py`, `tests/test_server_features.py`, `tests/test_time_filtering.py`, `src/cli_menu.py`, `src/clustering/clustering.py`, `src/clustering/dbscan_clustering.py`, `src/clustering/hdbscan_clustering.py`, `src/clustering/parallel_hdbscan_clustering.py`, `server.py`, `run.py`, `precompute_embeddings.py`, and many test files
- **Common issues**:
  - Line too long (C0301)
  - Missing docstrings (C0114, C0115, C0116)
  - Unused variables (W0612)
  - Wrong import order (C0411)
  - Too many arguments/local variables (R0913, R0914)
  - Too many branches/statements (R0912, R0915)
  - Trailing whitespace (C0303)
  - Broad exception catching (W0718)

### mypy
- **Files with issues**: 13 files
- **Common issues**:
  - Missing library stubs or py.typed marker for third-party libraries (import-untyped)
  - This is mostly due to missing type annotations for libraries like sklearn, hdbscan, shapely, networkx, etc.

## JavaScript Linter (eslint)
- **Files checked**: `static/app.js`
- **Results**: No errors reported

## CSS Linter (stylelint)
- **Files checked**: `static/style.css`
- **Results**: No errors reported

## Summary
- JavaScript and CSS files are clean
- Python files have some issues, mostly related to line length, trailing whitespace, and missing docstrings
- The syntax error in embedding_service.py is likely a false positive or parsing issue

## Recommendation
If we want to improve code quality further, we could:
1. Fix line length issues (increase the limit or break long lines)
2. Add docstrings to modules, classes, and functions
3. Fix import order issues
4. Remove unused variables
5. Add type annotations to untyped functions
