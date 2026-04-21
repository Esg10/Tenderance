from self_pruning_network_full import run_full_comparison, run_phase5
import pickle
import os

if __name__ == '__main__':
    all_results = run_full_comparison([1e-5, 1e-4, 1e-3], epochs=   1)
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    run_phase5(all_results, output_dir='outputs')
