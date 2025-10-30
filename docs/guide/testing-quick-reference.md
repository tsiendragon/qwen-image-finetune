# 测试快速参考指南

## 📋 快速开始

### 运行所有测试
```bash
pytest tests/ -v
```

### 按类型运行测试
```bash
# 仅运行单元测试（快速，无 I/O）
pytest tests/ -m "not integration and not e2e" -v

# 运行集成测试（需要文件 I/O）
pytest tests/ -m integration -v

# 运行端到端测试（需要 GPU）
pytest tests/ -m e2e -v

# 运行慢测试
pytest tests/ -m slow -v
```

### 按模块运行测试
```bash
# 测试特定模块
pytest tests/data/ -v
pytest tests/losses/ -v
pytest tests/models/ -v
pytest tests/trainer/ -v
pytest tests/scheduler/ -v
pytest tests/validation/ -v
pytest tests/utils/ -v

# 测试单个文件
pytest tests/data/test_cache_manager.py -v

# 测试单个函数
pytest tests/data/test_cache_manager.py::TestEmbeddingCacheManagerInit::test_initialization -v
```

### 查看覆盖率
```bash
# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=term-missing

# 生成 HTML 覆盖率报告
pytest tests/ --cov=src --cov-report=html
# 然后打开 htmlcov/index.html

# 检查特定模块覆盖率
pytest tests/ --cov=qflux.data --cov-report=term-missing
```

### 其他有用命令
```bash
# 查看所有可用标记
pytest --markers

# 显示最慢的 10 个测试
pytest tests/ --durations=10

# 失败时进入调试器
pytest tests/ --pdb

# 在第一个失败时停止
pytest tests/ -x

# 只运行上次失败的测试
pytest tests/ --lf

# 详细输出（包括 print 语句）
pytest tests/ -v -s
```

---

## 🏷️ 测试标记说明

### `@pytest.mark.integration`
需要本地依赖（文件 I/O、数据库等）的测试

**示例**:
```python
@pytest.mark.integration
def test_save_cache_to_disk(self, tmp_path):
    """Test saving cache to real filesystem"""
    ...
```

### `@pytest.mark.e2e`
需要外部服务（GPU、HuggingFace、网络等）的端到端测试

**示例**:
```python
@pytest.mark.e2e
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_train_with_real_model(self):
    """Test training with real model on GPU"""
    ...
```

### `@pytest.mark.slow`
运行时间较长（> 10秒）的测试

**示例**:
```python
@pytest.mark.slow
def test_full_training_epoch(self):
    """Test complete training epoch"""
    ...
```

---

## 🧪 测试编写最佳实践

### 1. 测试命名规范
```python
# ✅ 良好的命名
def test_process_image_returns_correct_shape(self):
    """Test that process_image returns expected shape"""
    ...

def test_cache_manager_handles_empty_file(self):
    """Test cache manager behavior with empty file"""
    ...

# ❌ 不好的命名
def test_1(self):
    ...

def test_function(self):
    ...
```

### 2. 使用 AAA 模式
```python
def test_example(self):
    """Test description"""
    # Arrange: 准备测试数据
    input_data = create_test_data()
    expected_output = calculate_expected()

    # Act: 执行被测试的操作
    result = function_under_test(input_data)

    # Assert: 验证结果
    assert result == expected_output
```

### 3. 使用 Fixtures
```python
# conftest.py 中定义
@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

# 测试中使用
def test_process_image(self, sample_image):
    """Test image processing"""
    result = process_image(sample_image)
    assert result.shape == (512, 512, 3)
```

### 4. 使用参数化测试
```python
@pytest.mark.parametrize("input,expected", [
    (0, 0),
    (1, 1),
    (2, 4),
    (3, 9),
])
def test_square_function(self, input, expected):
    """Test square function with various inputs"""
    assert square(input) == expected
```

### 5. 测试错误处理
```python
def test_function_raises_error_on_invalid_input(self):
    """Test that function raises ValueError for invalid input"""
    with pytest.raises(ValueError, match="输入不能为空"):
        function_under_test(None)
```

### 6. 使用临时文件
```python
def test_save_file(self, tmp_path):
    """Test file saving"""
    file_path = tmp_path / "output.txt"
    save_to_file("content", file_path)
    assert file_path.exists()
    assert file_path.read_text() == "content"
```

---

## 📝 常用 Fixtures

### `tmp_path`
提供临时目录（自动清理）
```python
def test_with_files(self, tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("content")
    ...
```

### `tmp_cache_dir`
提供测试用缓存目录（项目自定义）
```python
def test_cache(self, tmp_cache_dir):
    manager = EmbeddingCacheManager(str(tmp_cache_dir))
    ...
```

### `sample_image`
提供测试用 RGB 图像（项目自定义）
```python
def test_process(self, sample_image):
    result = process_image(sample_image)
    ...
```

### `mock_processor_config`
提供 mock 配置（项目自定义）
```python
def test_processor(self, mock_processor_config):
    processor = ImageProcessor(mock_processor_config)
    ...
```

---

## 🎯 测试覆盖目标

### 每个函数必测
- ✅ 正常输入/输出路径
- ✅ 边界条件（空值、极值）
- ✅ 错误处理（异常、无效输入）
- ✅ 类型验证

### 示例
```python
class TestImageProcessor:
    def test_process_image_normal(self):
        """Test normal image processing"""
        ...

    def test_process_image_empty_input(self):
        """Test with empty input (boundary)"""
        ...

    def test_process_image_invalid_format(self):
        """Test with invalid format (error)"""
        ...

    def test_process_image_wrong_type(self):
        """Test with wrong type (type check)"""
        ...
```

---

## 🚨 反模式（避免这些）

### ❌ 测试过于宽泛
```python
# 不好
def test_everything(self):
    # 测试 10 个不同的功能
    ...
```

### ❌ 无意义的命名
```python
# 不好
def test_1(self):
    ...
def test_function(self):
    ...
```

### ❌ 断言不具体
```python
# 不好
assert result  # 太模糊

# 好
assert result.status == "success"
assert len(result.items) == 5
```

### ❌ 单元测试中使用真实 I/O
```python
# 不好（在单元测试中）
def test_process(self):
    file = open("real_file.txt")  # ❌
    time.sleep(1)  # ❌
    requests.get("http://...")  # ❌
```

### ❌ 测试之间有依赖
```python
# 不好
class TestSuite:
    def test_step1(self):
        self.data = process()

    def test_step2(self):
        # 依赖 test_step1 ❌
        result = use(self.data)
```

---

## 📊 CI/CD 集成

### GitHub Actions 示例
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run unit tests
        run: pytest tests/ -m "not integration and not e2e" --cov=src

      - name: Run integration tests
        run: pytest tests/ -m integration

      - name: Check coverage
        run: pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80
```

---

## 🔍 调试技巧

### 使用 pdb 调试
```bash
# 失败时进入调试器
pytest tests/test_file.py --pdb

# 开始时就进入调试器
pytest tests/test_file.py --trace
```

### 查看详细输出
```bash
# 显示 print 输出
pytest tests/ -s

# 显示详细日志
pytest tests/ -v --log-cli-level=DEBUG
```

### 只运行特定测试
```bash
# 按名称匹配
pytest tests/ -k "cache"

# 排除某些测试
pytest tests/ -k "not slow"
```

---

## 📚 相关文档

- [PyTest 官方文档](https://docs.pytest.org/)
- [项目测试规范](.cursor/rules/pytest.mdc)
- [测试分析报告](docs/test-analysis-report.md)

---

**最后更新**: 2025-10-12
