import subprocess
import sys
def run_evaluation_suite():
    print("Starting RAG Evaluation Suite...")
    print("\nRunning Retrieval Evaluation...")
    subprocess.run([sys.executable,"-m","src.evaluation.evaluate_retrieval"],check=True)
    print("\nRunning Generation Evaluation...")
    subprocess.run([sys.executable,"-m","src.evaluation.evaluate_generation"],check=True)

    
if __name__=="__main__":
    run_evaluation_suite()