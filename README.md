# TransDreamer-AMLSim Baseline

This repository adapts **[TransDreamer](https://github.com/danijar/dreamerv2)**—a Transformer-based Reinforcement Learning World Model originally designed for image-based environments like Atari—to process **tabular financial transaction data** using the AMLSim dataset.

##  What We Did & Architecture

```mermaid
flowchart LR
    subgraph Data [Data Ingestion]
        direction TB
        DB[(AMLSim \n tx_log.csv)]:::db
        Env[AMLSimEnv \n Wrapper]:::env
        DB -->|Raw Features| Env
    end

    subgraph Encoder [Representation]
        direction TB
        State[7D State Vector]:::tensor
        TabEnc[Tabular Encoder MLP]:::net
        Env -->|Extracts| State
        State -->|Input| TabEnc
    end

    subgraph WM [Latent Dynamics]
        direction TB
        RSSM{Transformer \n RSSM Core}:::core
        TabEnc -->|d_model=600| RSSM
    end

    subgraph Decoders [Task Decoders]
        direction TB
        Rec[Tabular Decoder \n Observation Prior]:::dec
        Rew[Dense Decoder \n Anomaly/Reward]:::dec
        Act[Actor Decoder \n Policy]:::dec
        
        RSSM -->|State Recon| Rec
        RSSM -->|Fraud Prob| Rew
        RSSM -->|Allow/Block| Act
    end

    classDef db fill:#ffffff,stroke:#cbd5e1,stroke-width:2px,color:#334155,shape:cylinder;
    classDef env fill:#f8fafc,stroke:#94a3b8,stroke-width:2px,color:#0f172a,rx:4,ry:4;
    classDef tensor fill:#f1f5f9,stroke:#64748b,stroke-width:2px,color:#0f172a,rx:2,ry:2;
    classDef net fill:#e2e8f0,stroke:#475569,stroke-width:2px,color:#0f172a,rx:4,ry:4;
    classDef core fill:#334155,stroke:#0f172a,stroke-width:2px,color:#ffffff;
    classDef dec fill:#f8fafc,stroke:#94a3b8,stroke-width:2px,color:#0f172a,rx:4,ry:4;
```

TransDreamer natively expects 3D image tensors `(C, H, W)` and relies heavily on Convolutional Neural Networks (CNNs). To run it on tabular fraud data, we performed a "brain transplant" on the architecture:

1. **Custom Gym Environment (`envs/amlsim_env.py`)**:
   - Ingests AMLSim data and groups it by `nameOrig` to form transaction "trajectories" with long temporal horizons.
   - **State Space**: 7-dimensional tabular features (step, type, amount, old/new balances).
   - **Action Space**: Binary (0 = Allow, 1 = Block).
   - **Reward Logic**: `+1` for correct classification of `isSAR`, `-1` for incorrect.

2. **Tabular Encoders & Decoders (`model/modules_transformer.py`)**:
   - Bypassed the original `ImgEncoder` and `ImgDecoder`.
   - Built custom `TabularEncoder` and `TabularDecoder` MLPs capable of ingesting 1D tabular arrays and projecting them into the `d_model` dimensions required by the Transformer.

3. **Inference Pipeline (`evaluate_fraud.py`)**:
   - A custom evaluation script that loads trained checkpoints, feeds unseen sequences of account history, and predicts the "novel future state" to manually evaluate if the model is learning the structure of banking behavior over time.

### Temporal Trajectory Construction
To adapt static tabular logs for a Reinforcement Learning World Model, we fundamentally restructure the data into temporal sequences:

```mermaid
flowchart LR
    subgraph Raw [Static Dataset]
        direction TB
        R1[Transaction 1]:::raw
        R2[Transaction 2]:::raw
        R3[Transaction 3]:::raw
    end

    subgraph Process [Temporal Sequencing]
        direction TB
        G1{Group by \n nameOrig}:::proc
        G2[Sort by Time]:::proc
        G1 --> G2
    end

    subgraph Traj [Gym Environment Trajectories]
        direction TB
        T1([Account A: t=1 → t=2 → t=3]):::traj
        T2([Account B: t=1 → t=2]):::traj
    end
    
    Raw --> Process
    Process --> Traj

    classDef raw fill:#ffffff,stroke:#cbd5e1,stroke-width:2px,color:#334155,rx:2,ry:2;
    classDef proc fill:#f1f5f9,stroke:#64748b,stroke-width:2px,color:#0f172a;
    classDef traj fill:#334155,stroke:#0f172a,stroke-width:2px,color:#ffffff,rx:10,ry:10;
```

## 🔬 Experimental Findings

We trained this baseline on a Cloud GPU for 30,000 steps. Using the custom inference script (`evaluate_fraud.py`), we evaluated the model against unseen accounts and discovered two critical insights:

### 1. The Normalization Quirk
AMLSim's raw tabular features are currently being inadvertently scaled using TransDreamer's original Atari pixel normalization (`obs / 255.0 - 0.5`) before entering the `TabularEncoder`. Because our features are large monetary values rather than 0-255 pixels, this produces unusually large loss numbers. This quirk does not stop the model from learning (proving the robustness of the latent space), but fixing this should be prioritized in future architecture refinements.

### 2. Missing Arithmetic Constraints
When the model predicts future states, it predicts the transaction `Amount`, `Old Balance`, and `New Balance`. In reality, these should satisfy the equation `Old + Amount = New`. 

However, we found that the model's predictions **consistently violate this arithmetic by roughly $8 to $12** across all unseen accounts. 

**Why?** The architecture's `TabularDecoder` models each of these 7 tabular features as completely independent Gaussian distributions (`Independent(Normal(...))`). There is no structural joint constraint forcing them to align algebraically. The model must learn this arithmetic purely from data. 

#### Inference & Evaluation Flow
```mermaid
sequenceDiagram
    participant Env as AMLSimEnv
    participant Model as TransDreamer
    participant Dec as Tabular Decoder
    
    Env->>Model: Feed History (t=1 to 10)
    Model->>Dec: Combined Latent State (RNN + Prior)
    
    Note over Dec: Independent Gaussian Decoding
    Dec-->>Dec: Head 1: Amount
    Dec-->>Dec: Head 2: Old Balance
    Dec-->>Dec: Head 3: New Balance
    
    Dec->>Env: Predicted Novel State (t=11)
    
    Note over Env,Dec: Arithmetic Metric: |(Old + Amount) - New| != 0
```

**Conclusion:** Tracking the arithmetic error (`|Old + Amount - New|`) is a highly effective, novel metric to evaluate how well a World Model is learning the latent "rules" of a tabular environment, far beyond what standard loss curves can show.

## ⚙️ How to Run Evaluation

```bash
python evaluate_fraud.py model_000029001.pth
```
