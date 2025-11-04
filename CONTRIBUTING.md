# Contributing to WhoAmI

Thank you for your interest in contributing to WhoAmI! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/alanchelmickjr/whoami.git
   cd whoami
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install in development mode**
   ```bash
   pip install -e .
   ```

## Project Structure

```
whoami/
├── whoami/              # Main package
│   ├── __init__.py
│   ├── face_recognizer.py  # Core recognition logic
│   ├── gui.py              # GUI application
│   ├── cli.py              # CLI application
│   └── config.py           # Configuration management
├── examples/            # Example integrations
├── tests/              # Test files
├── run_gui.py          # GUI entry point
├── run_cli.py          # CLI entry point
└── setup.py            # Installation script
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep functions focused and modular
- Comment complex logic

## Adding Features

When adding new features:

1. **Keep it modular**: Add new functionality in separate modules
2. **Maintain backward compatibility**: Don't break existing APIs
3. **Update documentation**: Update README.md with new features
4. **Add examples**: Provide usage examples in the `examples/` directory
5. **Test your changes**: Ensure code works on both desktop and Jetson

## Testing

Run the structure validation test:
```bash
python tests/test_structure.py
```

Manual testing checklist:
- [ ] GUI starts without errors
- [ ] Camera connects successfully
- [ ] Face detection works
- [ ] Face recognition works
- [ ] Add/remove faces functionality
- [ ] CLI commands work
- [ ] Configuration loads/saves correctly

## Submitting Changes

1. Create a new branch for your feature
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Test thoroughly

4. Commit with clear messages
   ```bash
   git commit -m "Add feature: description of feature"
   ```

5. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include screenshots for UI changes
- Ensure all tests pass
- Keep changes focused (one feature per PR)

## Hardware Testing

If you have access to:
- **Oak D Series 3**: Test camera integration
- **Jetson Orin Nano**: Test performance and compatibility
- Other DepthAI devices: Test compatibility

## Feature Ideas

Some areas where contributions would be valuable:

- Multi-camera support
- Performance optimizations for Jetson
- REST API for remote access
- Integration with ROS
- Face detection confidence thresholds
- Database import/export functionality
- Additional camera settings controls
- Face tracking (temporal consistency)

## Questions?

If you have questions about contributing, please open an issue on GitHub.

## License

By contributing to WhoAmI, you agree that your contributions will be licensed under the same license as the project.
