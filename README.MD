# 🛡️ Codebase for FedEM: A Privacy-Preserving Framework for Concurrent Utility Preservation in Federated Learning

This repository contains the code for the paper:

**“FedEM: A Privacy-Preserving Framework for Concurrent Utility Preservation in Federated Learning”**  
[arXiv:2503.06021](https://arxiv.org/abs/2503.06021)

## 📁 Project Structure

```
├── README.md              # Project documentation
├── basic/                 # Common utility functions
├── data/                  # Dataset files
├── attack/                # Diverse DLG attack implementations
├── attacks/               # Perturbation generator
├── inversefed/            # Attack methods cloned from the InverseFed repo
├── fl/                    # FL simulation core
│   ├── client.py          # Federated client logic
│   └── server.py          # Federated server logic
├── experiments.py         # Main entry for launching experiments
├── main.py                # Internal single-experiment runner (used by experiments.py)
├── main_em.py             # Internal single-experiment runner (used by experiments.py)
└── result_ana.py          # Utilities for automatic result analysis
```

---

## 🚀 Getting Started

The main entry point is `main_em.py`. 

### 🔧 Example Usages

**Run a single experiment of FedSGD:

```bash
python main.py --dataset mnist  --gpu 0 **
```

**Run a single experiment of FedEM:

```bash
python main_em.py --dataset mnist  --gpu 0 
```

*notice: paths and directoies with the code should be manualy changed to ur own*

---

## 🔗 Citation

If you find this repo useful for your research or development, please cite our paper:

```
@article{xu2025fedem,
  title={FedEM: A Privacy-Preserving Framework for Concurrent Utility Preservation in Federated Learning},
  author={Xu, Mingcong and Zhang, Xiaojin and Chen, Wei and Jin, Hai},
  journal={arXiv preprint arXiv:2503.06021},
  year={2025}
}
```