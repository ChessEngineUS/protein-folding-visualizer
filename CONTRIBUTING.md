# Contributing to Protein Folding Visualizer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

### 1. Report Bugs 🐛

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU)
- Error messages and logs

### 2. Suggest Features ✨

We're particularly interested in:
- Additional benchmark datasets (CAMEO, CATH, Pfam)
- Novel uncertainty quantification methods
- Performance optimizations (quantization, distillation)
- Integration with experimental data (cryo-EM, X-ray)
- New visualization techniques
- Web interface development

### 3. Submit Pull Requests 🔧

#### Process

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/protein-folding-visualizer.git
   cd protein-folding-visualizer
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Follow code style guidelines (see below)
   - Add tests for new features
   - Update documentation

4. **Test your changes**
   ```bash
   pytest tests/
   black src/ tests/
   flake8 src/ tests/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add amazing feature: brief description"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/amazing-feature
   ```
   Then open a Pull Request on GitHub.

### 4. Improve Documentation 📚

Documentation improvements are always welcome:
- Fix typos or clarify explanations
- Add usage examples
- Create tutorials
- Translate documentation

## 📋 Code Style Guidelines

### Python Code

- **Formatting**: Use Black with default settings
  ```bash
  black src/ tests/
  ```

- **Linting**: Pass flake8 checks
  ```bash
  flake8 src/ tests/ --max-line-length=88
  ```

- **Type Hints**: Use type annotations
  ```python
  def predict(sequence: str, output_dir: str) -> PredictionResult:
      ...
  ```

- **Docstrings**: Use NumPy style
  ```python
  def function(param1: int, param2: str) -> bool:
      """
      Brief description.
      
      Parameters
      ----------
      param1 : int
          Description of param1
      param2 : str
          Description of param2
          
      Returns
      -------
      bool
          Description of return value
      """
      ...
  ```

### Notebooks

- Clear markdown descriptions
- Executable cells in order
- Output cleared before commit (except demo outputs)
- Colab compatibility tested

### Commit Messages

Follow conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat: add ensemble uncertainty quantification
fix: resolve CUDA memory leak in predictor
docs: update installation guide for Windows
test: add unit tests for affinity calculator
```

## 🧪 Testing Guidelines

### Unit Tests

Add tests for new features in `tests/`:

```python
import pytest
from src.module import function

def test_function_basic():
    """Test basic functionality."""
    result = function(input_data)
    assert result.is_valid()
    assert result.value > 0

def test_function_edge_cases():
    """Test edge cases."""
    with pytest.raises(ValueError):
        function(invalid_input)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_alphafold3.py

# Run specific test
pytest tests/test_alphafold3.py::test_validate_sequence
```

## 📊 Benchmark Contributions

If contributing benchmark results:

1. **Document methodology**
   - Dataset used
   - Model configuration
   - Hardware specifications
   - Random seeds

2. **Provide reproducibility**
   - Include configuration files
   - List all dependencies
   - Share preprocessing scripts

3. **Format results**
   - Use JSON or CSV
   - Include metadata
   - Provide summary statistics

## 🔬 Research Contributions

For novel research contributions:

1. **Add to appropriate module**
   - `src/evaluation/` for new metrics
   - `src/alphafold3/` or `src/boltz2/` for model improvements
   - `src/visualization/` for new plotting methods

2. **Document thoroughly**
   - Mathematical formulations
   - Algorithm descriptions
   - Complexity analysis
   - Citations to literature

3. **Validate rigorously**
   - Unit tests
   - Integration tests
   - Benchmark on standard datasets
   - Compare against baselines

## 📝 Documentation Standards

### API Documentation

```python
class Predictor:
    """
    AlphaFold 3 structure predictor.
    
    Implements diffusion-based architecture for biomolecular
    structure prediction with enhanced accuracy.
    
    Parameters
    ----------
    model_dir : str
        Path to model weights directory
    use_gpu : bool, optional
        Enable GPU acceleration (default: True)
        
    Attributes
    ----------
    model : torch.nn.Module
        Loaded neural network model
    config : dict
        Model configuration
        
    Examples
    --------
    >>> predictor = Predictor(model_dir='./models/af3')
    >>> result = predictor.predict('protein.fasta')
    >>> print(f"Mean pLDDT: {result.plddt.mean():.2f}")
    Mean pLDDT: 87.35
    """
```

### README Updates

When adding features:
- Update feature list
- Add usage example
- Update architecture diagram if needed
- Note any new dependencies

## 🤝 Code Review Process

### What We Look For

✅ **Good**
- Clear, readable code
- Comprehensive tests
- Updated documentation
- Follows style guidelines
- Addresses specific issue/feature

❌ **Avoid**
- Mixing multiple unrelated changes
- Breaking existing tests
- Missing documentation
- Uncommented complex code

### Review Timeline

- Initial response: 1-3 days
- Full review: 3-7 days
- Revisions: Iterative until approved

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

All contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in publications (for significant contributions)

## 💬 Getting Help

**Questions about contributing?**
- Open a discussion on GitHub
- Email the maintainers
- Join our community chat (coming soon)

## 🎓 First-Time Contributors

New to open source? Start with:
- Issues labeled `good-first-issue`
- Documentation improvements
- Adding examples
- Fixing typos

We're here to help! Don't hesitate to ask questions.

---

**Thank you for contributing to advancing protein structure prediction research! 🧬**
