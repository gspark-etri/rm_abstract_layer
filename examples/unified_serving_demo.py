"""
Unified Serving Demo

모든 서빙 엔진을 동일한 인터페이스로 사용하는 예제입니다.

Usage:
    python examples/unified_serving_demo.py
    python examples/unified_serving_demo.py --engine vllm
    python examples/unified_serving_demo.py --engine triton
    python examples/unified_serving_demo.py --engine torchserve
"""

import argparse
from rm_abstract.serving import (
    create_serving_engine,
    ServingConfig,
    ServingEngineType,
    DeviceTarget,
)


def demo_vllm():
    """vLLM 서빙 엔진 데모"""
    print("\n" + "=" * 60)
    print("  vLLM Serving Engine")
    print("=" * 60)
    
    config = ServingConfig(
        engine=ServingEngineType.VLLM,
        device=DeviceTarget.GPU,
        model_name="gpt2",
    )
    
    # 동일한 인터페이스로 사용
    with create_serving_engine(config) as engine:
        print(f"\n  Engine: {engine.name}")
        print(f"  Loading model...")
        
        engine.load_model("gpt2")
        
        prompt = "The future of AI is"
        print(f"\n  [Input]  \"{prompt}\"")
        
        output = engine.infer(prompt, max_tokens=30)
        print(f"  [Output] \"{output}\"")
    
    print("\n  ✓ vLLM demo completed!")
    return True


def demo_triton():
    """Triton 서빙 엔진 데모 (자동 Docker 관리)"""
    print("\n" + "=" * 60)
    print("  Triton Serving Engine (Auto Docker)")
    print("=" * 60)
    
    config = ServingConfig(
        engine=ServingEngineType.TRITON,
        device=DeviceTarget.GPU,
        model_name="gpt2_triton",
        port=8000,
    )
    
    try:
        # 동일한 인터페이스로 사용
        # context manager가 자동으로 Docker 시작/종료
        with create_serving_engine(config) as engine:
            print(f"\n  Engine: {engine.name}")
            print(f"  Docker container starting...")
            
            engine.load_model("gpt2")
            
            prompt = "Hello, I am"
            print(f"\n  [Input]  \"{prompt}\"")
            
            output = engine.infer(prompt)
            print(f"  [Output] \"{output}\"")
        
        print("\n  ✓ Triton demo completed!")
        return True
        
    except Exception as e:
        print(f"\n  ⚠ Triton demo skipped: {e}")
        print("  Note: Requires Docker and nvidia-container-toolkit")
        return False


def demo_torchserve():
    """TorchServe 서빙 엔진 데모 (자동 서버 관리)"""
    print("\n" + "=" * 60)
    print("  TorchServe Serving Engine (Auto Server)")
    print("=" * 60)
    
    config = ServingConfig(
        engine=ServingEngineType.TORCHSERVE,
        device=DeviceTarget.GPU,
        model_name="gpt2_ts",
        port=8080,
    )
    
    try:
        # 동일한 인터페이스로 사용
        # context manager가 자동으로 서버 시작/종료
        with create_serving_engine(config) as engine:
            print(f"\n  Engine: {engine.name}")
            print(f"  Server starting...")
            
            engine.load_model("gpt2")
            
            prompt = "Machine learning is"
            print(f"\n  [Input]  \"{prompt}\"")
            
            output = engine.infer(prompt)
            print(f"  [Output] \"{output}\"")
        
        print("\n  ✓ TorchServe demo completed!")
        return True
        
    except Exception as e:
        print(f"\n  ⚠ TorchServe demo skipped: {e}")
        print("  Note: Requires Java 11+ and torchserve package")
        return False


def demo_unified_interface():
    """통합 인터페이스 데모"""
    print("\n" + "=" * 60)
    print("  Unified Interface Demo")
    print("=" * 60)
    
    print("""
  모든 엔진이 동일한 인터페이스를 사용합니다:

  ```python
  with create_serving_engine(config) as engine:
      engine.load_model("gpt2")
      output = engine.infer("Hello")
  ```

  - vLLM: 라이브러리로 직접 실행
  - Triton: Docker 자동 시작/종료
  - TorchServe: 서버 프로세스 자동 시작/종료
    """)


def main():
    parser = argparse.ArgumentParser(description="Unified Serving Demo")
    parser.add_argument(
        "--engine",
        choices=["vllm", "triton", "torchserve", "all"],
        default="vllm",
        help="Which engine to demo (default: vllm)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Unified Serving Demo - RM Abstract Layer")
    print("=" * 60)
    
    demo_unified_interface()
    
    results = {}
    
    if args.engine in ["vllm", "all"]:
        results["vLLM"] = demo_vllm()
    
    if args.engine in ["triton", "all"]:
        results["Triton"] = demo_triton()
    
    if args.engine in ["torchserve", "all"]:
        results["TorchServe"] = demo_torchserve()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "✓" if success else "⚠"
        print(f"  {status} {name}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

