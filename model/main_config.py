import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model.model import UndirectedContrastiveClusteringModel
from verifydataset import verify_dataset_quality

from util.DatasetGennerate import ClusteringDataset, MeaningfulClusteringDataset, SemanticClusteringDataset, get_dataset

def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    epoch_losses = {'total': [], 'edge': [], 'contrast': []}
    
    for node_features, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False):
        node_features = node_features.to(device)
        labels = labels.to(device)
        
        # 前向传播
        edge_scores, loss_dict = model(node_features, labels, epoch)
        
        # 反向传播
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 记录损失
        epoch_losses['total'].append(loss_dict['total_loss'].item())
        epoch_losses['edge'].append(loss_dict['loss_edge'].item())
        if 'loss_contrast' in loss_dict:
            epoch_losses['contrast'].append(loss_dict['loss_contrast'].item())
    
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
    return avg_losses


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for node_features, labels in dataloader:
            node_features = node_features.to(device)
            labels = labels.to(device)
            
            edge_scores, _ = model(node_features)
            
            # 转换为0/1预测
            preds = (edge_scores > 0.5).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算指标
    accuracy = (all_preds == all_labels).float().mean().item()
    
    # F1 score
    tp = ((all_preds == 0) & (all_labels == 0)).sum().float()
    fp = ((all_preds == 0) & (all_labels == 1)).sum().float()
    fn = ((all_preds == 1) & (all_labels == 0)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 总损失
    axes[0, 0].plot(history['train_total_loss'], label='Train', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 边分类损失
    axes[0, 1].plot(history['train_edge_loss'], label='Train', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Edge Loss', fontsize=12)
    axes[0, 1].set_title('Edge Classification Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 对比学习损失
    if history['train_contrast_loss']:
        axes[1, 0].plot(history['train_contrast_loss'], label='Train', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Contrastive Loss', fontsize=12)
        axes[1, 0].set_title('Contrastive Learning Loss', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Contrastive Loss', ha='center', va='center', fontsize=14)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
    
    # 准确率和F1
    epochs_val = [i * 5 for i in range(len(history['val_precision']))]
    axes[1, 1].plot(epochs_val, history['val_precision'], label='Precision', marker='o', linewidth=2)
    axes[1, 1].plot(epochs_val, history['val_f1'], label='F1 Score', marker='s', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 训练曲线已保存为 {save_path}")


# ==================== 第八部分：主程序 ====================

def main(args):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 配置
    config = {
        
        'n_clusters': 2,     # 集合数量
        # 'hidden_dim': 64,
        # 'n_gcn_layers': 4,
        # 'temperature': 0.07,
        # 'use_contrastive': True,
        # 'batch_size': 32,
        # 'n_epochs': 200,
        # 'lr': 0.001,
        'train_samples': 1000,
        'val_samples': 200,
        'test_samples': 200,
        # 'SizeofDataset': 5000,
    }
    
    print("="*80)
    print("集合划分问题 - 图对比学习模型（最终版）")
    print("="*80)
    print(f"✓ 产品数: {args.get('m', '未设置')}, 工序数: {args.get('n', '未设置')}")
    print(f"✓ 节点数: {args.get('n_nodes', '未设置')}, 批次数：{args.get('batch_size', '未设置')}")
    print(f"✓ 隐藏维度: {args.get('hidden_dim', '未设置')}, GCN层数: {args.get('n_gcn_layers', '未设置')}")
    print(f"✓ 使用对比学习: {args.get('use_contrastive', '未设置')}")
    print(f"✓ 边类型: 无向边")
    print(f"✓ 损失计算: 论文方法（节点级Softmax + NLLLoss + 类别权重）")
    print("="*80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}\n")
    
    # 创建数据集
    print("创建数据集...")
    # train_dataset = MeaningfulClusteringDataset(
    #     config['train_samples'], args.m, args.n, 
    #     args.n_nodes, config['n_clusters']
    # )
    # val_dataset = MeaningfulClusteringDataset(
    #     config['val_samples'], args.m, args.n, 
    #     args.n_nodes, config['n_clusters']
    # )
    # test_dataset = MeaningfulClusteringDataset(
    #     config['test_samples'], args.m, args.n, 
    #     args.n_nodes, config['n_clusters']
    # )
    # # 验证数据集质量
    # print("验证数据集质量...")
    # verify_dataset_quality(train_dataset)
    # verify_dataset_quality(val_dataset)
    # verify_dataset_quality(test_dataset)
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    train_loader, val_loader, test_loader = get_dataset(args, SizeofDataset = args.get('SizeofDataset', 5000))
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    args_n = args.get('n_batch', 0) * 4
    # 创建模型
    print("创建模型...")
    model = UndirectedContrastiveClusteringModel(
        m=args.get('m', 0),
        n=args_n,
        n_nodes=args.get('n_nodes', 0),
        hidden_dim=args.get('hidden_dim', 0),
        n_gcn_layers=args.get('n_gcn_layers', 0),
        temperature=args.get('temperature', 0),
        use_contrastive=args.get('use_contrastive', False)
    ).to(device)
    
    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 模型参数量: {n_params:,}\n")
    
    # 验证对称性
    print("验证模型输出的对称性...")
    with torch.no_grad():
        test_input = torch.randn(2, args.get('n_nodes', 0), args.get('m', 0) * args_n).to(device)
        test_output, _ = model(test_input)
        max_asymmetry = (test_output - test_output.transpose(1, 2)).abs().max().item()
        print(f"✓ 最大不对称性: {max_asymmetry:.10f} (应接近0)\n")
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.get('lr', 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.get('n_epochs', 0))
    
    # 训练历史
    history = {
        'train_total_loss': [],
        'train_edge_loss': [],
        'train_contrast_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
    }
    
    best_val_f1 = 0.0
    best_val_pre = 0.0
    
    # 训练循环
    print("开始训练...")
    print("="*80)
    
    for epoch in range(args.get('n_epochs', 0)):
        # 训练
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        
        # 记录训练损失
        history['train_total_loss'].append(train_losses['total'])
        history['train_edge_loss'].append(train_losses['edge'])
        if train_losses['contrast'] > 0:
            history['train_contrast_loss'].append(train_losses['contrast'])
        
        # 验证
        if (epoch + 1) % 2 == 0:
            val_metrics = evaluate(model, val_loader, device)
            
            # 记录验证指标
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1'])
            
            print(f"\nEpoch {epoch+1}/{args.get('n_epochs', 0)}")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"Edge: {train_losses['edge']:.4f}, "
                  f"Contrast: {train_losses['contrast']:.4f}")
            print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, "
                  f"Prec: {val_metrics['precision']:.4f}, "
                  f"Rec: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics['precision'] > best_val_pre:
                best_val_pre = val_metrics['precision']
                model_name = f"seruconfig_W{args.get('n_nodes', 0)}_J{args.get('n_batch', 0)}_pre.pt"
                torch.save(model.state_dict(), model_name)
                print(f"  ✓ 保存最佳模型 (Prec: {best_val_pre:.4f})")
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                model_name = f"seruconfig_W{args.get('n_nodes', 0)}_J{args.get('n_batch', 0)}_f1.pt"
                torch.save(model.state_dict(), model_name)
                print(f"  ✓ 保存最佳模型 (F1: {best_val_f1:.4f})")
    
    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    
    # 加载最佳模型进行测试
    print("\n在测试集上评估...")
    model.load_state_dict(torch.load(model_name))
    test_metrics = evaluate(model, test_loader, device)
    
    print("\n测试集结果:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_curves(history,save_path=f"training_curves_W{args.get('n_nodes', 0)}_J{args.get('n_batch', 0)}.png")
    
    print("\n" + "="*80)
    print("全部完成!")
    print("="*80)
    
    return model, history, test_metrics

# def parse_args():
#     parser = argparse.ArgumentParser()
#     # 高准确率的数据集
#     # parser.add_argument("--data_dir", type=str, default=r"C:\code\datasets\Seru_datasets\processed\randomT50-60_WAll_Exemples_5000\processed\data_randomT50-60_WAll_Exemples.pt", help="数据文件路径")
    
#     # parser.add_argument("--data_dir", type=str, default=r"C:\code\datasets\Seru_datasets\processed\randomT50-60_W7_WAll_Exemples_2000\processed\data_randomT50-60_W7_WAll_Exemples_2000.pt", help="数据文件路径")
#     parser.add_argument("--data_dir", type=str, default=r"C:\code\datasets\Seru_datasets\processed\randomT50-60_W7_WAll_Exemples_2000\processed\data_randomT50-60_W7_WAll_Exemples_2000.pt", help="数据文件路径")
    
#     parser.add_argument("--n_nodes", type=int, default=7)
#     parser.add_argument("--m", type=int, default=1)
#     parser.add_argument("--n", type=int, default=35)
#     parser.add_argument("--SizeofDataset", type=int, default=5000)

#     parser.add_argument("--test_size", type=float, default=0.2)
#     parser.add_argument("--batch_size", type=int, default=32)

#     parser.add_argument("--hidden_dim", type=int, default=128)
#     parser.add_argument("--n_gcn_layers", type=int, default=4)
#     parser.add_argument("--temperature", type=float, default=0.07)
#     parser.add_argument("--use_contrastive", type=bool, default=True)
#     parser.add_argument("--n_epochs", type=int, default=200)

#     parser.add_argument("--lr", type=float, default=0.001)
    
    
    
#     return parser.parse_args()

from util.config_loader import load_yaml_config

if __name__ == "__main__":
    # args = parse_args()
    args = load_yaml_config('JCompany_W6_J10_5000.yaml')
    model, history, test_metrics = main(args)
