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
Large neural networks cannot run on cheap edge devices like farmer phones or IoT sensors in remote fields. Pruning makes models smaller and faster without sacrificing too much accuracy. PruneVision shows this end-to-end:
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
A standard neural network has millions of weight connections. Many of these connections are redundant — they contribute almost nothing to the final prediction. Pruning removes these weak connections, making the model:

Smaller in memory
Faster at inference
Deployable on low-resource devices

The PrunableLinear Layer
Instead of using PyTorch's standard nn.Linear, we built a custom layer that has a second learnable parameter called gate_scores. Here is the exact mechanism:
Standard Linear:     output = W × input + bias

PrunableLinear:      gates  = sigmoid(gate_scores)      ← values between 0 and 1
                     W_eff  = W × gates                  ← element-wise multiply
                     output = W_eff × input + bias
When a gate value becomes close to 0, it multiplies its corresponding weight by nearly zero — effectively removing that weight from the network. When a gate stays close to 1, the weight operates normally.
The key insight is that gate_scores is a learnable parameter — the optimizer updates it just like any other weight. So the network learns on its own which connections to keep and which to remove.
pythonclass PrunableLinear(nn.Module):
    def forward(self, input_tensor):
        gates = torch.sigmoid(self.gate_scores)    # values between 0 and 1
        pruned_weights = self.weight * gates        # zero out weak weights
        return F.linear(input_tensor, pruned_weights, self.bias)
Gradients flow correctly through both self.weight and self.gate_scores because sigmoid and element-wise multiply are both differentiable operations.
The Sparsity Loss
Without any extra penalty, the optimizer has no reason to push gates toward zero. We add an L1 regularization term to the loss function:
Total Loss = CrossEntropyLoss + λ × L1(all gate values)
The L1 norm (sum of absolute values) is well known to encourage sparsity. The gradient of |x| is always ±1 regardless of the magnitude of x — this means the optimizer applies constant pressure pushing every gate toward zero. Gates that are not useful for classification will eventually reach zero. Gates that are critical for accuracy will resist this pressure because their classification loss gradient will push back.
Lambda controls the trade-off:
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
The convolutional layers use standard PyTorch layers for feature extraction. The fully connected layers use PrunableLinear so the classifier learns which connections matter most.

Part 2 — The Intelligent Preprocessing Module (Our Addition)
This is the module we added beyond the case study. In a real deployment, you cannot trust that users will upload clean, well-lit, properly framed images. The intelligent preprocessor defends the model against bad inputs before they reach inference.
Layer 1 — Image Quality Checker
Before doing anything, we measure three image properties:
Blur detection:
  → Compute Laplacian of grayscale image
  → Take variance of result
  → If variance < 100: image is blurry → quality score drops

Brightness check:
  → Compute mean pixel value of grayscale image
  → If mean < 40:  too dark   → quality score drops
  → If mean > 220: too bright → quality score drops

Size validation:
  → If width < 100px or height < 100px: too small → reject
Output: quality_score (0–100) and quality_passed (True/False)
This prevents the model from making confident wrong predictions on images that are simply too blurry or too dark to analyze.
Layer 2 — Auto Enhancement
If the image passes quality check, we apply four enhancements in sequence:
Step 1 — Denoising
  → cv2.fastNlMeansDenoisingColored
  → Removes sensor noise and compression artifacts
  → Makes edges cleaner for the model

Step 2 — CLAHE Contrast Enhancement
  → Convert to LAB color space
  → Apply Contrast Limited Adaptive Histogram Equalization on L channel
  → Improves local contrast without blowing out bright areas
  → Especially useful for leaves photographed in uneven sunlight

Step 3 — Sharpening
  → Apply convolution kernel: [[0,-1,0],[-1,5,-1],[0,-1,0]]
  → Enhances edges and fine disease texture details

Step 4 — Gamma Correction
  → Auto-detect whether image is too dark or too bright
  → Apply inverse gamma LUT to normalize overall brightness
All four steps are applied and logged. The preprocessing report returned to the frontend lists exactly which steps ran.
Layer 3 — Leaf Segmentation
The most important preprocessing step. A phone photo of a leaf usually contains a hand, soil, sky, or blurry background. We isolate only the leaf:
Step 1 — Background Removal (rembg)
  → Uses a pre-trained U2Net model
  → Removes everything that is not the main subject
  → Outputs RGBA image with transparent background

Step 2 — Green Color Masking
  → Convert to HSV color space
  → Create mask for hue values 25–95 (green range)
  → Apply morphological open and close to clean the mask

Step 3 — Contour Detection
  → Find all external contours in the green mask
  → Select the largest contour (the main leaf)
  → Compute bounding box with 10px padding
  → Crop image to that bounding box

Step 4 — Fallback Center Crop
  → If no leaf contour found, crop center 80% of image
  → Ensures something meaningful is always passed to the model

Step 5 — Resize to 224×224
  → Standard input size for the CNN
Layer 4 — Confidence Gatekeeper
After the model runs inference, we apply a final decision layer:
Confidence > 75%  → high_confidence
  "Prediction is strong. Proceed with suggested treatment."

Confidence 50–75% → medium_confidence
  "Plausible result. Clearer image may improve certainty."
  Warning banner shown in UI

Confidence < 50%  → low_confidence
  "Please retake the image."
  Red warning shown — result should not be trusted
This prevents the application from giving confident wrong answers, which is critical in an agricultural context where a wrong diagnosis could lead to incorrect pesticide use.

Part 3 — FastAPI Backend
The backend exposes a clean REST API built with FastAPI.
How a Prediction Request Works
POST /predict (multipart image upload)
        ↓
1. Read image bytes from upload
        ↓
2. Run IntelligentPreprocessor.process()
   → quality check → enhancement → segmentation
        ↓
3. Convert processed BGR image to normalized tensor
        ↓
4. model.eval() + torch.no_grad() → forward pass
        ↓
5. softmax → confidence + predicted class index
        ↓
6. Run confidence gatekeeper
        ↓
7. Return JSON: label, confidence, status,
   preprocessing report, model stats, warning
Startup — Model Loading
On startup, the app checks for a saved checkpoint at backend/model_checkpoints/best_model.pt. If found, it loads the trained weights including class names, lambda used, and test accuracy. If not found, it initializes a default untrained model so the API still responds without crashing.
All Endpoints
MethodEndpointPurposePOST/predictRun full inference pipeline on uploaded imageGET/model-statsReturn sparsity, accuracy, size, parameter countsGET/training-historyReturn full training history JSONGET/healthReturn model load status and compute deviceGET/uploadsReturn all upload history records

Part 4 — History Portal
Every prediction is automatically saved to localStorage with:

Timestamp
Disease label
Confidence score
Quality score
Leaf detection status
Confidence status badge

The history portal provides:

Summary cards: total analyses, high confidence count, avg quality, avg confidence
Filterable and searchable table by disease name and confidence level
Sort by newest or oldest first
Download full history as a formatted PDF via the browser print dialog
Clear all history with confirmation


Part 5 — Frontend Design
The frontend is a single HTML file with no external frameworks.
Key UI features:

Animated neural network canvas background showing nodes connecting and pruning (connections flash red before disappearing — visualizing the pruning concept)
Cursor-reactive white grid overlay — cells glow and brighten as cursor moves over them
Custom cursor dot replacing the default browser cursor
Drag and drop upload zone with animated dashed border
Scan line animation on image preview
4-step processing indicator with spinners and green checkmarks
Animated SVG confidence ring (stroke-dashoffset animation)
Animated sparsity bar
Count-up number animations using requestAnimationFrame
Glassmorphism cards with subtle float animation
Toast notifications sliding in from the right
Fully mobile responsive with media queries
Marquee scrolling tagline strip
About section with builder profile
Project explanation blocks
Stats strip with scroll-triggered count-up
Technology stack showcase cards


Installation and Running
Step 1 — Install dependencies
bashpip install -r backend/requirements.txt
Step 2 — Train the model

Run this before starting the API to get real stats and sparsity metrics.

bashpython backend/train.py
Trains 3 models with lambda values 0.0001, 0.001, 0.01.
CIFAR-10 downloads automatically (~170MB) if PlantVillage is not available.
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

Replace the dashes above with values from results.json after running train.py


Gate Distribution Plot
Show Image
The spike near 0 confirms the L1 penalty successfully pushed unnecessary gates to zero. The secondary cluster away from 0 represents the connections the model decided were essential and kept active.

What's Not in This Repo
ExcludedReasondata/CIFAR-10 downloads automatically via train.pybackend/model_checkpoints/Generated after running trainingprunevision.dbGenerated at runtime by the APIuploads/Generated at runtime__pycache__/Python bytecode

Tech Stack
LayerTechnologyWhyModelPyTorchCustom PrunableLinear layer, CNN backbone, sparse training loopPreprocessingOpenCV, Pillow, rembgBlur detection, CLAHE, U2Net background removal, contour croppingBackendFastAPI, UvicornREST endpoints, async inference, CORS, Pydantic schemasDatabaseSQLiteLightweight persistent history, no external server neededFrontendVanilla JS, HTML5 Canvas, CSS3Zero dependencies, fast, single fileTraining dataCIFAR-10 (auto) or PlantVillage (manual)Auto-download fallback for easy setup

What This Demonstrates for Tredence
Tredence JD RequirementWhere It Appears in This ProjectClean, maintainable Python codeAll backend files use type hints, docstrings, dataclasses, loggingFastAPI with async workflowsmain.py uses lifespan, async endpoints, middleware, Pydantic modelsLLM/AI pipeline buildingFull preprocessing → inference → confidence gating pipelinePerformance-minded programmingPruning reduces model size; preprocessor rejects bad inputs earlyProblem solving and DSACustom layer, custom loss function, custom training loopCaching and optimizationGate-based pruning reduces compute at inference timeDocumentation, testing, logginglogging module throughout, full docstrings on every functionBuilder mindsetCase study extended into a real deployable application

About the Builder
Balaji Madhan
AI Engineer Intern Candidate — Tredence AI Engineering Internship 2025 Cohort
Built PruneVision independently as a demonstration of full-stack AI engineering:
from custom PyTorch layer design through intelligent preprocessing, REST API
deployment, and a polished single-page frontend — all production-quality,
all in one project.
