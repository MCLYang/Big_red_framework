import wandb
wandb.init(project="preemptable", resume=True)
package = wandb.restore('best_model.pth',run_path = 'gufengy/point_ring_light_vs_pointnet5c_simpledataset/tn6zxc7c')

print(package)

