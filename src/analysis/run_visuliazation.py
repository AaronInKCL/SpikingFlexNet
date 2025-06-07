import torch
import os
import matplotlib.pyplot as plt
from run_loader import load_model_and_data
from gradient_flow import plot_gradient_flow
from hessian import compute_hessian_spectrum
from intermediate_plot import plot_intermediate_representations
from loss_surface import compute_loss_surface

def run_analysis_suite(checkpoint_path, dataset_name, save_dir='./analysis_results'):
    os.makedirs(save_dir, exist_ok=True)

    # Load model and data
    model, train_loader, test_loader = load_model_and_data(checkpoint_path, dataset_name)
    model.eval()

    # 1. Plot Gradient Flow
    print("\n[1/5] Plotting Gradient Flow...")
    plot_gradient_flow(model, save_path=os.path.join(save_dir, 'gradient_flow.png'))

    # 2. Compute Hessian Spectrum
    print("\n[2/5] Computing Hessian Spectrum...")
    compute_hessian_spectrum(model, train_loader, save_dir=save_dir)

    # 3. Plot Intermediate Representations
    print("\n[3/5] Plotting Intermediate Representations...")
    plot_intermediate_representations(model, save_path=os.path.join(save_dir, 'intermediate_representations.png'))

    # 4. Compute Loss Surface (3D Plot)
    print("\n[4/5] Computing Loss Surface and plotting 3D Surface...")
    X, Y, Z = compute_loss_surface(model, test_loader, distance=1.0, steps=30)

    # Save 3D Loss Surface Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('3D Loss Surface')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_surface_3d.png'), dpi=300)
    plt.close(fig)

    print(f"3D Loss Surface saved to {save_dir}/loss_surface_3d.png")

    # 5. (Optional) Add other analyses if needed
    print("\n[5/5] Analysis Completed!")

if __name__ == '__main__':
    checkpoint = './checkpoints/model_best.pth'  # Example checkpoint
    dataset = 'cifar10'  # Example dataset
    run_analysis_suite(checkpoint, dataset)
