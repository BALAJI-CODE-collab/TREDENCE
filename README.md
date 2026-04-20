PruneVision — Intelligent Plant Disease Detector

Built by Balaji Madhan as part of the Tredence AI Engineering Internship application.
This project goes beyond the case study requirement — combining a self-pruning neural network
with a real-world plant disease detection application, intelligent preprocessing pipeline,
and a production-ready FastAPI backend.


Why This Project Goes Beyond the Case Study
The original case study asked for:

A prunable neural network trained on CIFAR-10
A sparsity loss with multiple lambda values
A report comparing accuracy vs sparsity

We built all of that — and wrapped it inside a real-world plant disease detection application to demonstrate how self-pruning neural networks solve an actual deployment problem.
The Real-World Problem We Solved
Large neural networks cannot run on cheap edge devices like farmer phones or IoT sensors in remote fields. Pruning makes models smaller and faster without sacrificing too much accuracy. PruneVision shows this end to end:
Farmer uploads leaf photo
          ↓
Intelligent preprocessor cleans the image
          ↓
Self-pruning model (30–70% weights removed) runs inference
          ↓
Confidence-aware result with recommendation
          ↓
History saved for future reference + PDF export

Full Architecture
PruneVision/
├── backend/
│   ├── main.py              → FastAPI app, endpoints, model loading, inference
│   ├── model.py             → PrunableLinear layer + PrunableCNN + training helpers
│   ├── preprocessor.py      → 4-layer intelligent preprocessing pipeline
│   ├── train.py             → Multi-lambda training loop, evaluation, plot generation
│   ├── database.py          → SQLite history persistence
│   └── requirements.txt
├── frontend/
│   └── index.html           → Full SPA with animations, history portal, PDF export
├── results.json             → Lambda comparison results (generated after training)
├── gate_distribution_best.png  → Gate value histogram (proof of pruning)
├── report.md                → Case study report with analysis
└── README.md

Part 1 — The Self-Pruning Neural Network (Case Study Core)
What is Pruning?
A standard neural network has millions of weight connections. Many of these connections are redundant — they contribute almost nothing to the final prediction. Pruning removes these weak connections, making the model smaller in memory, faster at inference, and deployable on low-resource devices.
The PrunableLinear Layer
Instead of using PyTorch's standard nn.Linear, we built a custom layer that has a second learnable parameter called gate_scores. Here is the exact mechanism:
Standard Linear:     output = W × input + bias

PrunableLinear:      gates  = sigmoid(gate_scores)      ← values between 0 and 1
                     W_eff  = W × gates                  ← element-wise multiply
                     output = W_eff × input + bias
When a gate value becomes close to 0, it multiplies its corresponding weight by nearly zero — effectively removing that weight from the network. The key insight is that gate_scores is a learnable parameter — the optimizer updates it just like any other weight. So the network learns on its own which connections to keep and which to remove.
pythonclass PrunableLinear(nn.Module):
    def forward(self, input_tensor):
        gates = torch.sigmoid(self.gate_scores)    # values between 0 and 1
        pruned_weights = self.weight * gates        # zero out weak weights
        return F.linear(input_tensor, pruned_weights, self.bias)
Gradients flow correctly through both self.weight and self.gate_scores because sigmoid and element-wise multiply are both differentiable operations.
The Sparsity Loss
Without any extra penalty, the optimizer has no reason to push gates toward zero. We add an L1 regularization term:
Total Loss = CrossEntropyLoss + λ × L1(all gate values)
The gradient of |x| is always ±1 regardless of the magnitude of x — this means the optimizer applies constant pressure pushing every gate toward zero. Gates that are not useful for classification will eventually reach zero. Gates that are critical for accuracy will resist this pressure because their classification loss gradient will push back.
LambdaEffectLow (0.0001)Light pruning, high accuracy preservedMid (0.001)Balanced sparsity and accuracyHigh (0.01)Aggressive pruning, accuracy may drop
Network Architecture
Input image (3 × 224 × 224)
        ↓
Conv2d 3→32  + BatchNorm + ReLU + MaxPool
        ↓
Conv2d 32→64 + BatchNorm + ReLU + MaxPool
        ↓
Conv2d 64→128 + BatchNorm + ReLU + MaxPool
        ↓
AdaptiveAvgPool → 128 × 4 × 4 = 2048 features
        ↓
PrunableLinear 2048 → 512   ← gates here
        ↓
PrunableLinear 512  → 256   ← gates here
        ↓
PrunableLinear 256  → num_classes   ← gates here
        ↓
Prediction

Part 2 — The Intelligent Preprocessing Module (Our Addition)
This is the module we added beyond the case study. In a real deployment, you cannot trust that users will upload clean, well-lit, properly framed images. The intelligent preprocessor defends the model against bad inputs before they reach inference.
Layer 1 — Image Quality Checker
Blur detection:
  → Compute Laplacian of grayscale image
  → Take variance of result
  → If variance < 100: image is blurry → quality score drops

Brightness check:
  → If mean pixel < 40:  too dark   → quality score drops
  → If mean pixel > 220: too bright → quality score drops

Size validation:
  → If width < 100px or height < 100px → reject
Layer 2 — Auto Enhancement
Step 1 — Denoising
  → cv2.fastNlMeansDenoisingColored
  → Removes sensor noise and compression artifacts

Step 2 — CLAHE Contrast Enhancement
  → Convert to LAB color space
  → Apply CLAHE on L channel
  → Improves local contrast without blowing out bright areas

Step 3 — Sharpening
  → Kernel: [[0,-1,0],[-1,5,-1],[0,-1,0]]
  → Enhances edges and fine disease texture details

Step 4 — Gamma Correction
  → Auto-detect brightness level
  → Apply inverse gamma LUT to normalize brightness
Layer 3 — Leaf Segmentation
Step 1 — Background Removal (rembg U2Net)
  → Removes everything that is not the main subject

Step 2 — Green Color Masking
  → HSV mask for hue values 25–95
  → Morphological open and close to clean mask

Step 3 — Contour Detection
  → Select largest contour (the main leaf)
  → Crop bounding box with 10px padding

Step 4 — Fallback Center Crop
  → If no leaf found, crop center 80% of image

Step 5 — Resize to 224×224
Layer 4 — Confidence Gatekeeper
Confidence > 75%  → high_confidence
  "Prediction is strong. Proceed with suggested treatment."

Confidence 50–75% → medium_confidence
  "Plausible result. Clearer image may improve certainty."

Confidence < 50%  → low_confidence
  "Please retake the image."

Part 3 — FastAPI Backend
How a Prediction Request Works
POST /predict
        ↓
1. Read image bytes from upload
        ↓
2. Run IntelligentPreprocessor.process()
   → quality check → enhancement → segmentation
        ↓
3. Convert processed image to normalized tensor
        ↓
4. model.eval() + torch.no_grad() → forward pass
        ↓
5. softmax → confidence + predicted class index
        ↓
6. Run confidence gatekeeper
        ↓
7. Return JSON response
All Endpoints
MethodEndpointPurposePOST/predictRun full inference pipeline on uploaded imageGET/model-statsReturn sparsity, accuracy, size, parameter countsGET/training-historyReturn full training history JSONGET/healthReturn model load status and compute deviceGET/uploadsReturn all upload history records

Part 4 — History Portal
Every prediction is automatically saved to localStorage with timestamp, disease label, confidence score, quality score, leaf detection status, and confidence badge. The portal provides a searchable and filterable table, summary stat cards, sort by newest or oldest, and a one-click PDF download via the browser print dialog.

Part 5 — Frontend Design
The frontend is a single HTML file with no external frameworks. Key features include an animated neural network canvas background, cursor-reactive white grid overlay where cells glow as the cursor moves, custom cursor dot, drag and drop upload with animated dashed border, scan line animation on image preview, 4-step processing indicator, animated SVG confidence ring, animated sparsity bar, count-up number animations, glassmorphism cards with float animation, toast notifications, marquee scrolling tagline strip, about section with builder profile, project explanation blocks, stats strip with scroll-triggered animations, and technology stack showcase cards.

Installation and Running
Step 1 — Install dependencies
bashpip install -r backend/requirements.txt
Step 2 — Train the model

Must run before starting the API to get real sparsity and accuracy metrics.

bashpython backend/train.py
Trains 3 models with lambda values 0.0001, 0.001, 0.01. CIFAR-10 downloads automatically (~170MB) if PlantVillage is not available.
Saves:

results.json
backend/training_history.json
backend/model_checkpoints/best_model.pt
gate_distribution_best.png

Step 3 — Start the API
bashpython -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
Step 4 — Open the frontend
Open frontend/index.html in browser or visit http://localhost:8000

Results
LambdaTest AccuracySparsity Level %0.0001——0.001——0.01——

Replace dashes with values from results.json after running train.py


Gate Distribution Plot
Show Image
The spike near 0 confirms the L1 penalty successfully pushed unnecessary gates to zero. The secondary cluster away from 0 represents connections the model decided were essential and kept active.

What Is Not in This Repo
ExcludedReasondata/CIFAR-10 downloads automatically via train.pybackend/model_checkpoints/Generated after running trainingprunevision.dbGenerated at runtimeuploads/Generated at runtime__pycache__/Python bytecode

Tech Stack
LayerTechnologyWhyModelPyTorchCustom PrunableLinear, CNN backbone, sparse trainingPreprocessingOpenCV, Pillow, rembgBlur detection, CLAHE, U2Net background removalBackendFastAPI, UvicornAsync endpoints, CORS, Pydantic schemasDatabaseSQLiteLightweight persistent history, no external serverFrontendVanilla JS, HTML5 Canvas, CSS3Zero dependencies, fast, single fileTraining dataCIFAR-10 (auto) or PlantVillage (manual)Auto-download fallback for easy setup

What This Demonstrates for Tredence
Tredence JD RequirementWhere It AppearsClean maintainable Python codeType hints, docstrings, dataclasses, logging throughoutFastAPI with async workflowsLifespan, async endpoints, middleware, Pydantic modelsAI pipeline buildingPreprocessing → inference → confidence gating pipelinePerformance-minded programmingPruning reduces model size, preprocessor rejects bad inputs earlyProblem solving and DSACustom layer, custom loss, custom training loopDocumentation and logginglogging module throughout, full docstrings on every functionBuilder mindsetCase study extended into a real deployable application

About the Builder
Balaji Madhan
AI Engineer Intern Candidate — Tredence AI Engineering Internship 2025 Cohort
Built PruneVision independently as a demonstration of full-stack AI engineering — from custom PyTorch layer design through intelligent preprocessing, REST API deployment, and a polished single-page frontend. All production-quality, all in one project.
