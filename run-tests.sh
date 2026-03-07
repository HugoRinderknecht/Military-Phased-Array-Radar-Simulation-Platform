#!/bin/bash
# 测试运行脚本

set -e

echo "========================================"
echo "雷达仿真平台测试套件"
echo "========================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 函数：打印信息
print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# 函数：打印成功
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 函数：打印错误
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# 解析参数
TEST_TYPE="${1:-all}"
VERBOSE="${2:-false}"

# 设置详细输出选项
PYTEST_OPTS="-v"
if [ "$VERBOSE" = "false" ]; then
    PYTEST_OPTS="--tb=short"
fi

# ============================================
# 后端测试
# ============================================

run_backend_tests() {
    print_info "运行后端测试..."

    cd "$BACKEND_DIR"

    # 检查Python环境
    if ! command -v python &> /dev/null; then
        print_error "Python未安装或不在PATH中"
        return 1
    fi

    # 检查依赖
    print_info "检查Python依赖..."
    pip install -q -r requirements.txt || {
        print_error "安装Python依赖失败"
        return 1
    }

    # 运行不同类型的测试
    case "$TEST_TYPE" in
        unit)
            print_info "运行单元测试..."
            pytest tests/unit/ $PYTEST_OPTS
            ;;
        integration)
            print_info "运行集成测试..."
            pytest tests/integration/ $PYTEST_OPTS
            ;;
        e2e)
            print_info "运行端到端测试..."
            pytest tests/e2e/ $PYTEST_OPTS
            ;;
        backend)
            print_info "运行所有后端测试..."
            pytest tests/ $PYTEST_OPTS --cov=app --cov-report=term-missing --cov-report=html
            ;;
        all)
            print_info "运行所有后端测试..."
            pytest tests/ $PYTEST_OPTS --cov=app --cov-report=term-missing --cov-report=html
            ;;
    esac

    if [ $? -eq 0 ]; then
        print_success "后端测试通过"
        return 0
    else
        print_error "后端测试失败"
        return 1
    fi
}

# ============================================
# 前端测试
# ============================================

run_frontend_tests() {
    print_info "运行前端测试..."

    cd "$FRONTEND_DIR"

    # 检查Node.js环境
    if ! command -v node &> /dev/null; then
        print_error "Node.js未安装或不在PATH中"
        return 1
    fi

    # 检查依赖
    if [ ! -d "node_modules" ]; then
        print_info "安装npm依赖..."
        npm install || {
            print_error "安装npm依赖失败"
            return 1
        }
    fi

    # 运行测试
    case "$TEST_TYPE" in
        unit)
            print_info "运行前端单元测试..."
            npm run test:run -- --reporter=verbose
            ;;
        coverage)
            print_info "运行前端测试覆盖率..."
            npm run test:coverage
            ;;
        frontend)
            print_info "运行所有前端测试..."
            npm run test:run -- --reporter=verbose
            ;;
        all)
            print_info "运行所有前端测试..."
            npm run test:run -- --reporter=verbose
            ;;
    esac

    if [ $? -eq 0 ]; then
        print_success "前端测试通过"
        return 0
    else
        print_error "前端测试失败"
        return 1
    fi
}

# ============================================
# 主逻辑
# ============================================

print_info "测试类型: $TEST_TYPE"

BACKEND_FAILED=0
FRONTEND_FAILED=0

# 根据测试类型运行相应的测试
case "$TEST_TYPE" in
    unit|integration|e2e|backend)
        run_backend_tests
        BACKEND_FAILED=$?
        ;;
    unit|coverage|frontend)
        run_frontend_tests
        FRONTEND_FAILED=$?
        ;;
    all)
        run_backend_tests
        BACKEND_FAILED=$?

        echo ""
        run_frontend_tests
        FRONTEND_FAILED=$?
        ;;
    *)
        print_error "未知测试类型: $TEST_TYPE"
        echo "可用选项: all, unit, integration, e2e, backend, frontend, coverage"
        exit 1
        ;;
esac

# 汇总结果
echo ""
echo "========================================"
echo "测试结果汇总"
echo "========================================"

if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "backend" ] || [ "$TEST_TYPE" in "unit|integration|e2e" ]; then
    if [ $BACKEND_FAILED -eq 0 ]; then
        print_success "后端测试: 通过"
    else
        print_error "后端测试: 失败"
    fi
fi

if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "frontend" ] || [ "$TEST_TYPE" = "coverage" ]; then
    if [ $FRONTEND_FAILED -eq 0 ]; then
        print_success "前端测试: 通过"
    else
        print_error "前端测试: 失败"
    fi
fi

echo "========================================"

# 返回适当的退出代码
if [ $BACKEND_FAILED -ne 0 ] || [ $FRONTEND_FAILED -ne 0 ]; then
    exit 1
else
    print_success "所有测试通过!"
    exit 0
fi
