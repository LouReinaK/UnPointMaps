"""
Performance tests for hull computation algorithm.

This module contains performance benchmarks to measure and optimize
the hull computation algorithm's execution time and memory usage.
"""

import time
import numpy as np
import psutil
import os
from typing import List, Dict, Optional
import logging
from src.utils.hull_logic import get_alpha_shape, compute_cluster_hulls

# Configure logging
logger = logging.getLogger('hull_performance')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class HullPerformanceBenchmark:
    """Benchmark suite for hull computation performance."""

    def __init__(self):
        self.results = []
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def generate_test_points(self, n_points: int,
                             seed: int = 42) -> List[List[float]]:
        """Generate random test points for benchmarking."""
        np.random.seed(seed)
        # Generate points in a unit square
        points = np.random.rand(n_points, 2)
        return points.tolist()

    def generate_clustered_points(self,
                                  n_clusters: int,
                                  points_per_cluster: int,
                                  seed: int = 42) -> List[List[List[float]]]:
        """Generate clustered test data."""
        np.random.seed(seed)
        clusters = []

        for i in range(n_clusters):
            # Random center for each cluster
            center = np.random.rand(2)
            # Generate points around center
            cluster_points = center + \
                np.random.randn(points_per_cluster, 2) * 0.1
            clusters.append(cluster_points.tolist())

        return clusters

    def benchmark_single_hull(self, n_points: int, alpha: float = 1.0,
                              auto_alpha_quantile: float = 0.95) -> Dict:
        """Benchmark single hull computation."""
        logger.info(f"Benchmarking single hull with {n_points} points...")

        points = self.generate_test_points(n_points)

        # Warm up cache
        _ = get_alpha_shape(points[:10], alpha=alpha,
                            auto_alpha_quantile=auto_alpha_quantile)

        # Measure memory before
        mem_before = self.get_memory_usage()

        # Measure time
        start_time = time.time()
        result = get_alpha_shape(
            points,
            alpha=alpha,
            auto_alpha_quantile=auto_alpha_quantile)
        end_time = time.time()

        # Measure memory after
        mem_after = self.get_memory_usage()

        execution_time = end_time - start_time
        memory_delta = mem_after - mem_before

        logger.info(
            f"  Time: {execution_time:.4f}s, Memory: {memory_delta:.2f}MB")

        return {
            'test_type': 'single_hull',
            'n_points': n_points,
            'alpha': alpha,
            'auto_alpha_quantile': auto_alpha_quantile,
            'execution_time': execution_time,
            'memory_delta_mb': memory_delta,
            'result_size': len(result) if result else 0
        }

    def benchmark_batch_hulls(
            self,
            n_clusters: int,
            points_per_cluster: int,
            alpha: float = 1.0,
            auto_alpha_quantile: float = 0.95,
            max_workers: Optional[int] = None) -> Dict:
        """Benchmark batch hull computation."""
        logger.info(
            f"Benchmarking batch hulls: {n_clusters} clusters, {points_per_cluster} points each...")

        clusters = self.generate_clustered_points(
            n_clusters, points_per_cluster)

        # Warm up cache
        _ = compute_cluster_hulls(
            clusters[:2], alpha=alpha, auto_alpha_quantile=auto_alpha_quantile)

        # Measure memory before
        mem_before = self.get_memory_usage()

        # Measure time
        start_time = time.time()
        results = compute_cluster_hulls(
            clusters,
            alpha=alpha,
            auto_alpha_quantile=auto_alpha_quantile,
            max_workers=max_workers)
        end_time = time.time()

        # Measure memory after
        mem_after = self.get_memory_usage()

        execution_time = end_time - start_time
        memory_delta = mem_after - mem_before

        logger.info(
            f"  Time: {execution_time:.4f}s, Memory: {memory_delta:.2f}MB")

        return {
            'test_type': 'batch_hulls',
            'n_clusters': n_clusters,
            'points_per_cluster': points_per_cluster,
            'total_points': n_clusters * points_per_cluster,
            'alpha': alpha,
            'auto_alpha_quantile': auto_alpha_quantile,
            'max_workers': max_workers,
            'execution_time': execution_time,
            'memory_delta_mb': memory_delta,
            'result_size': len(results) if results else 0
        }

    def benchmark_cache_effectiveness(
            self,
            n_points: int,
            iterations: int = 5) -> Dict:
        """Benchmark cache effectiveness by running same computation multiple times."""
        logger.info(
            f"Benchmarking cache effectiveness with {n_points} points, {iterations} iterations...")

        points = self.generate_test_points(n_points)

        times = []
        for i in range(iterations):
            start_time = time.time()
            get_alpha_shape(points, alpha=1.0, auto_alpha_quantile=0.95)
            end_time = time.time()
            times.append(end_time - start_time)
            logger.info(f"  Iteration {i+1}: {times[-1]:.4f}s")

        # Calculate speedup from cache
        if len(times) > 1:
            speedup = times[0] / times[-1] if times[-1] > 0 else 1.0
        else:
            speedup = 1.0

        logger.info(f"  Cache speedup: {speedup:.2f}x")

        return {
            'test_type': 'cache_effectiveness',
            'n_points': n_points,
            'iterations': iterations,
            'times': times,
            'speedup': speedup,
            'avg_time': np.mean(times),
            'std_time': np.std(times)
        }

    def benchmark_scaling(
            self,
            point_counts: List[int],
            alpha: float = 1.0) -> List[Dict]:
        """Benchmark scaling with different input sizes."""
        logger.info(f"Benchmarking scaling with point counts: {point_counts}")

        results = []
        for n_points in point_counts:
            result = self.benchmark_single_hull(n_points, alpha=alpha)
            results.append(result)

        # Calculate scaling factor
        if len(results) > 1:
            times = [r['execution_time'] for r in results]
            scaling_factors = []
            for i in range(1, len(times)):
                if times[i - 1] > 0:
                    factor = times[i] / times[i - 1]
                    scaling_factors.append(factor)
                else:
                    scaling_factors.append(0)

            logger.info(f"  Scaling factors: {scaling_factors}")
            logger.info(
                f"  Average scaling factor: {np.mean(scaling_factors):.2f}x")

        return results

    def benchmark_alpha_sensitivity(
            self,
            n_points: int,
            alpha_values: List[float]) -> List[Dict]:
        """Benchmark sensitivity to alpha parameter."""
        logger.info(
            f"Benchmarking alpha sensitivity with {n_points} points, alpha values: {alpha_values}")

        results = []
        for alpha in alpha_values:
            result = self.benchmark_single_hull(n_points, alpha=alpha)
            results.append(result)

        return results

    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark suite."""
        logger.info("=" * 60)
        logger.info("Starting Comprehensive Hull Performance Benchmark")
        logger.info("=" * 60)

        all_results = []

        # Test 1: Single hull with varying sizes
        logger.info("\n--- Test 1: Single Hull Scaling ---")
        point_counts = [100, 500, 1000, 2000, 5000]
        scaling_results = self.benchmark_scaling(point_counts)
        all_results.extend(scaling_results)

        # Test 2: Batch hulls with varying cluster counts
        logger.info("\n--- Test 2: Batch Hulls Scaling ---")
        batch_configs = [
            (10, 50),   # 10 clusters, 50 points each
            (20, 50),   # 20 clusters, 50 points each
            (50, 50),   # 50 clusters, 50 points each
            (100, 50),  # 100 clusters, 50 points each
        ]

        for n_clusters, points_per_cluster in batch_configs:
            result = self.benchmark_batch_hulls(n_clusters, points_per_cluster)
            all_results.append(result)

        # Test 3: Cache effectiveness
        logger.info("\n--- Test 3: Cache Effectiveness ---")
        cache_result = self.benchmark_cache_effectiveness(1000, iterations=5)
        all_results.append(cache_result)

        # Test 4: Alpha sensitivity
        logger.info("\n--- Test 4: Alpha Sensitivity ---")
        alpha_values = [0.5, 1.0, 2.0, 5.0]
        alpha_results = self.benchmark_alpha_sensitivity(1000, alpha_values)
        all_results.extend(alpha_results)

        # Test 5: Parallel execution
        logger.info("\n--- Test 5: Parallel Execution ---")
        parallel_configs = [
            (50, 100, 1),   # 50 clusters, 100 points, 1 worker
            (50, 100, 2),   # 50 clusters, 100 points, 2 workers
            (50, 100, 4),   # 50 clusters, 100 points, 4 workers
            (50, 100, 8),   # 50 clusters, 100 points, 8 workers
        ]

        for n_clusters, points_per_cluster, max_workers in parallel_configs:
            result = self.benchmark_batch_hulls(
                n_clusters, points_per_cluster, max_workers=max_workers)
            all_results.append(result)

        # Generate summary
        summary = self._generate_summary(all_results)

        logger.info("\n" + "=" * 60)
        logger.info("Benchmark Summary")
        logger.info("=" * 60)
        logger.info(summary)

        return {
            'summary': summary,
            'detailed_results': all_results
        }

    def _generate_summary(self, results: List[Dict]) -> str:
        """Generate a summary of benchmark results."""
        if not results:
            return "No results to summarize."

        summary_lines = []

        # Group by test type
        test_types: Dict[str, List[Dict]] = {}
        for result in results:
            test_type = result.get('test_type', 'unknown')
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)

        for test_type, test_results in test_types.items():
            summary_lines.append(f"\n{test_type.upper()}:")

            if test_type == 'single_hull':
                for r in test_results:
                    summary_lines.append(
                        f"  {r['n_points']} points: {r['execution_time']:.4f}s")

            elif test_type == 'batch_hulls':
                for r in test_results:
                    summary_lines.append(
                        f"  {r['n_clusters']} clusters ({r['total_points']} total points): {r['execution_time']:.4f}s")

            elif test_type == 'cache_effectiveness':
                for r in test_results:
                    summary_lines.append(
                        f"  {r['n_points']} points, {r['iterations']} iterations: {r['speedup']:.2f}x speedup")

            elif test_type == 'alpha_sensitivity':
                for r in test_results:
                    summary_lines.append(
                        f"  alpha={r['alpha']}: {r['execution_time']:.4f}s")

            elif test_type == 'parallel_execution':
                for r in test_results:
                    summary_lines.append(
                        f"  {r['n_clusters']} clusters, {r['max_workers']} workers: {r['execution_time']:.4f}s")

        return "\n".join(summary_lines)


def main():
    """Main entry point for performance testing."""
    benchmark = HullPerformanceBenchmark()

    try:
        results = benchmark.run_comprehensive_benchmark()

        # Save results to file
        import json
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"hull_performance_benchmark_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {output_file}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
