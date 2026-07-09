# TransDreamer-PaySim Baseline

This repository adapts **[TransDreamer](https://github.com/danijar/dreamerv2)**—a Transformer-based Reinforcement Learning World Model originally designed for image-based environments like Atari—to process **tabular financial transaction data** using the [PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1).

## 🚀 What We Did

TransDreamer natively expects 3D image tensors `(C, H, W)` and relies heavily on Convolutional Neural Networks (CNNs). To run it on tabular fraud data, we essentially performed a "brain transplant" on the architecture:

1. **Custom Gym Environment (`envs/paysim_env.py`)**:
   - Ingests `paysim.csv` and groups the data by `nameOrig` to form transaction "trajectories".
   - **State Space**: 7-dimensional tabular features (step, type, amount, old/new balances).
   - **Action Space**: Binary (0 = Allow, 1 = Block).
   - **Reward Logic**: `+1` for correct classification of `isFraud`, `-1` for incorrect.

2. **Tabular Encoders & Decoders (`model/modules_transformer.py`)**:
   - Bypassed the original `ImgEncoder` and `ImgDecoder`.
   - Built custom `TabularEncoder` and `TabularDecoder` MLPs capable of ingesting 1D tabular arrays and projecting them into the `d_model` dimensions required by the Transformer.

3. **Config Hacking (`config_files/configs_paysim.py` & `envs/__init__.py`)**:
   - Re-routed the environment initialization to trick the Atari-centric pipeline into accepting a dummy name (`paysim_dummy`).
   - Disabled TensorBoard video summaries (since our tabular states cannot be visualized as MP4s).
   - Solved Numpy deprecation errors (`np.float`) caused by modern dependencies.

## ⚙️ How to Run

1. Clone the repository and ensure you have `paysim.csv` in the parent directory (or update the path in `configs_paysim.py`).
2. Activate the conda environment:
   ```bash
   conda activate trans_dreamer
   ```
3. Run the training script:
   ```bash
   python main.py --config-file config_files/configs_paysim.py
   ```

## 🔬 Research & Dataset Limitations

> **Note:** This codebase successfully proves the *structural integration* of a World Model on tabular financial data, but PaySim itself is fundamentally misaligned with long-horizon sequence modeling.

**Trajectory Analysis:**
* Total trajectories (by `nameOrig`): ~6.35 Million
* Mean length: `1.001`
* Median length: `1.0`

Because the fraud simulated in PaySim typically occurs in depth-2 chains (Transfer → Cash Out), grouping by `nameOrig` destroys the temporal tracking of the *funds*. Even if graph-walks were used to construct trajectories, the chains remain too short to justify complex Recurrent State Space Models (RSSM) or Transformers, which excel at solving the **long-term credit assignment problem**.

This repository serves as a baseline boilerplate for adapting World Models to tabular ML, but for impactful financial sequence modeling, a dataset with richer, longer-horizon temporal dynamics (like HFT order books or dynamic supply chains) is recommended.