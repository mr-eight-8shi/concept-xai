import argparse
import logging
import os
import sys

from .com import AppConfig
from .core import MNISTPipeline
from .core import GeometricPipeline


def action_runner():
  """Decorator to manage common runtime initialization, configuration injection, 
     logging, and graceful exception handling for pipeline executions.
  """
  def decorator(func):
    def wrapper(*args, **kwargs):
      config = AppConfig()
      logger = config.logger
      try:
        return func(config, logger, *args, **kwargs)
      except KeyboardInterrupt:
        msg = "[ABORT] Process interrupted by user."
        logger.warning(msg)
        print(f"\n\n{msg}\n")

    return wrapper

  return decorator


@action_runner()
def run_geometric_train(config, logger):
  """Executes the pre-training pipeline for the geometric feature filter and concept dictionary."""
  logger.info("=== [Geometric Feature Model Training] Started ===")
  pipeline = GeometricPipeline(device=config.device, config=config)
  pipeline.train()
  logger.info("=== [Geometric Feature Model Training] Completed ===")


@action_runner()
def run_mnist_train(config, logger):
  """Executes the training pipeline for the MNIST spatial concept classification model."""
  logger.info("=== [MNIST Classification Model Training] Started ===")
  pipeline = MNISTPipeline(device=config.device, config=config)
  pipeline.train()
  logger.info("=== [MNIST Classification Model Training] Completed ===")


@action_runner()
def run_mnist_test(config, logger):
  """Executes the evaluation test pipeline on the MNIST test dataset."""
  logger.info("=== [MNIST Classification Model Testing] Started ===")
  pipeline = MNISTPipeline(device=config.device, config=config)
  pipeline.test()
  logger.info("=== [MNIST Classification Model Testing] Completed ===")


@action_runner()
def run_train_data_eval(config, logger):
  """Executes XAI evaluation on training data, mapping gradients and exporting concept JSONs/grids."""
  logger.info("=== [Training Data Evaluation & Anomaly Extraction] Started ===")
  pipeline = MNISTPipeline(device=config.device, config=config)
  pipeline.train_data_eval()
  logger.info("=== [Training Data Evaluation & Anomaly Extraction] Completed ===")


@action_runner()
def run_model_init(config, logger):
  """Resets and deletes all saved model checkpoints and cache files from the workspace directory."""
  logger.info("=== Resetting checkpoints and caches ===")
  space_dir = getattr(config, "space_dir", "space")
  
  target_files = [
      "checkpoint_geometric_filter.pth",
      "checkpoint_concept_dictionary.pth",
      "checkpoint_mnist_spatial_nn.pth",
      "concept_labels.json",
  ]
  
  for fname in target_files:
    path = os.path.join(space_dir, fname)
    if os.path.exists(path):
      os.remove(path)
      logger.info(f"Removed: {path}")
  print("\n✅ Initialization completed successfully.")


def interactive_menu():
  """Displays an interactive CLI menu loop for step-by-step pipeline execution and verification."""
  while True:
    print("\n" + "=" * 65)
    print("    Concept XAI - MNIST Step-by-Step Verification System")
    print("=" * 65)
    print("    [1] Train Geometric Feature & Dictionary Models")
    print("    [2] Train MNIST Spatial Concept Classifier")
    print("    [3] Test MNIST Classification Model")
    print("    [4] Evaluate Training Data & Extract Label/Concept Anomalies")
    print("    ---------------------------------------------------------")
    print("    [9] Init - Reset All Checkpoints & Caches")
    print("    [q] Quit - Exit Application")
    print("=" * 65)

    try:
      choice = input("Enter choice: ").strip().lower()
    except KeyboardInterrupt:
      print("\nExiting. Thank you!")
      break

    if choice in ["q", "quit"]:
      print("Exiting. Thank you!")
      break
    elif choice in ["1"]:
      run_geometric_train()
    elif choice in ["2"]:
      run_mnist_train()
    elif choice in ["3"]:
      run_mnist_test()
    elif choice in ["4"]:
      run_train_data_eval()
    elif choice in ["9", "init"]:
      confirm = (
          input(
              "⚠️ All saved checkpoints and caches will be deleted. Are you sure? (y/n): "
          )
          .strip()
          .lower()
      )
      if confirm == "y":
        run_model_init()
      else:
        print("Initialization cancelled.")
    else:
      print("[ERROR] Invalid choice. Please try again.")


def cli():
  """Parses command-line arguments to execute a specific action directly or launch the interactive menu."""
  parser = argparse.ArgumentParser(
      description="Modular Concept Extraction System for MNIST"
  )
  parser.add_argument("action", nargs="?", help="Action to execute (1, 2, 3, 4, init, etc.)")
  args = parser.parse_args()

  try:
    action = args.action.lower() if args.action else ""
    if action in ["1"]:
      run_geometric_train()
    elif action in ["2"]:
      run_mnist_train()
    elif action in ["3"]:
      run_mnist_test()
    elif action in ["4"]:
      run_train_data_eval()
    elif action in ["9", "init"]:
      run_model_init()
    else:
      interactive_menu()
  except KeyboardInterrupt:
    print("\n\n[ABORT] Program terminated.\n")
    sys.exit(0)


if __name__ == "__main__":
  cli()