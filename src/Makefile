CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -pthread
BUILD_DIR = ../build
HEADERS = ring_buffer.h importance_metrics.h metrics.h signal_loader.h rpeak_eval.h

all: sweep cross stats spectral stress

# Main sweep
sweep: $(BUILD_DIR)/run_sweep
$(BUILD_DIR)/run_sweep: main.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ main.cpp
	@echo "Build complete: $@"

# Cross-domain validation (N5)
cross: $(BUILD_DIR)/run_cross_domain
$(BUILD_DIR)/run_cross_domain: cross_domain.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ cross_domain.cpp
	@echo "Build complete: $@"

# Statistical analyzer (reads existing CSVs, no re-run)
stats: $(BUILD_DIR)/stats_analyzer
$(BUILD_DIR)/stats_analyzer: stats_analyzer.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ stats_analyzer.cpp
	@echo "Build complete: $@"

# Spectral mechanism analysis
spectral: $(BUILD_DIR)/spectral_analysis
$(BUILD_DIR)/spectral_analysis: spectral_analysis.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ spectral_analysis.cpp
	@echo "Build complete: $@"

# High-compression stress test
stress: $(BUILD_DIR)/stress_test
$(BUILD_DIR)/stress_test: stress_test.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ stress_test.cpp
	@echo "Build complete: $@"

# Latency benchmark
latency: $(BUILD_DIR)/latency_bench
$(BUILD_DIR)/latency_bench: latency_bench.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ latency_bench.cpp
	@echo "Build complete: $@"

# Sensitivity sweep
sensitivity: $(BUILD_DIR)/sensitivity_sweep
$(BUILD_DIR)/sensitivity_sweep: sensitivity_sweep.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ sensitivity_sweep.cpp
	@echo "Build complete: $@"

# N2 validation
n2: $(BUILD_DIR)/n2_validate
$(BUILD_DIR)/n2_validate: n2_validate.cpp ring_buffer_v2.h
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ n2_validate.cpp
	@echo "Build complete: $@"

# ============================================================
# Run targets
# ============================================================

# STEP 1: Stats on existing data (instant, no experiments)
run-stats: stats
	cd $(BUILD_DIR) && ./stats_analyzer

# STEP 2: Spectral mechanism (moderate time, vibration only)
run-spectral: spectral
	cd $(BUILD_DIR) && ./spectral_analysis

# STEP 3: Stress test quick (fast preview)
run-stress-quick: stress
	cd $(BUILD_DIR) && ./stress_test --quick

# STEP 3b: Stress test full (long run)
run-stress: stress
	cd $(BUILD_DIR) && ./stress_test

# Original runs
quick: sweep
	cd $(BUILD_DIR) && ./run_sweep --quick
cross-quick: cross
	cd $(BUILD_DIR) && ./run_cross_domain --quick
run: sweep
	cd $(BUILD_DIR) && ./run_sweep
cross-run: cross
	cd $(BUILD_DIR) && ./run_cross_domain

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all sweep cross stats spectral stress latency sensitivity n2 \
        run-stats run-spectral run-stress-quick run-stress \
        quick cross-quick run cross-run clean
