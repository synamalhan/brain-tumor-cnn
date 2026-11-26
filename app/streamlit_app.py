import json
import os
import sys

import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.resnet_classifier import build_resnet_classifier, grad_cam  # noqa: E402
from utils.preprocessing import split_dataset  # noqa: E402

CLS_WEIGHTS = os.path.join(ROOT, 'saved_models/resnet_classifier.weights.h5')
UNET_PATH = os.path.join(ROOT, 'saved_models/unet_model.keras')
RESULTS = os.path.join(ROOT, 'docs/results')
SAMPLES = os.path.join(ROOT, 'data/download/Brain Tumor Data Set/Brain Tumor Data Set')
LGG = os.path.join(ROOT, 'data/lgg/kaggle_3m')

st.set_page_config(page_title='NeuroScan · Brain Tumor AI', page_icon='🧠', layout='wide')
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1250px;}
  .hero {background: linear-gradient(120deg, #312e81 0%, #6d28d9 55%, #db2777 100%);
         padding: 1.8rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;}
  .hero h1 {color: white; margin: 0; font-size: 2.2rem;}
  .hero p {opacity: .85; margin: .4rem 0 0 0; font-size: 1.05rem;}
  .verdict {border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; border: 1px solid;}
  .verdict.tumor {background: rgba(219,39,119,.08); border-color: rgba(219,39,119,.45);}
  .verdict.healthy {background: rgba(16,185,129,.08); border-color: rgba(16,185,129,.45);}
  .verdict .label {font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; opacity: .7;}
  .verdict .value {font-size: 1.45rem; font-weight: 700; margin-top: .2rem;}
  .caption {text-align: center; opacity: .7; font-size: .9rem; margin-top: .3rem;}
  [data-testid="stMetricValue"] {font-size: 1.4rem;}
  [data-testid="stMetric"] {background: rgba(109,40,217,.05); border: 1px solid rgba(109,40,217,.15);
                            border-radius: 12px; padding: .7rem 1rem;}
  .stTabs [data-baseweb="tab"] {font-size: 1.05rem; padding: .6rem 1.2rem;}
</style>
<div class="hero">
  <h1>🧠 NeuroScan</h1>
  <p>Brain MRI tumor detection with a fine-tuned ResNet50 + Grad-CAM, and tumor segmentation with a U-Net.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    clf = None
    if os.path.exists(CLS_WEIGHTS):
        clf = build_resnet_classifier(weights=None)
        clf.load_weights(CLS_WEIGHTS)
    unet = tf.keras.models.load_model(UNET_PATH, compile=False) if os.path.exists(UNET_PATH) else None
    return clf, unet


@st.cache_data
def lgg_test_patients():
    """Same patient-level split as scripts/train_unet.py, computed from file counts only (no image loading)."""
    patients = sorted(p for p in os.listdir(LGG) if os.path.isdir(os.path.join(LGG, p)))
    tifs = {p: sorted(f for f in os.listdir(os.path.join(LGG, p)) if f.endswith('.tif') and not f.endswith('_mask.tif'))
            for p in patients}
    groups = np.concatenate([[i] * len(tifs[p]) for i, p in enumerate(patients)])
    idx = np.arange(len(groups))
    _, _, (test_idx, _) = split_dataset(idx, idx, groups, test_size=0.1, val_size=0.1)
    # slices in anatomical order for the slider
    return {patients[g]: sorted(tifs[patients[g]], key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
            for g in sorted(set(groups[test_idx]))}


def load_metrics(name):
    path = os.path.join(RESULTS, name)
    return json.load(open(path)) if os.path.exists(path) else None


def overlay(img, heat, alpha, cmap='inferno'):
    heat = np.asarray(Image.fromarray(np.uint8(heat * 255)).resize(img.size, Image.BILINEAR)) / 255.0
    colored = matplotlib.colormaps[cmap](heat)[..., :3] * 255
    a = alpha if cmap == 'inferno' else alpha * (heat[..., None] > 0)  # masks: tint only masked pixels
    return Image.fromarray(np.uint8((1 - a) * np.asarray(img) + a * colored))


def show_images(views):
    for col, (im, cap) in zip(st.columns(len(views)), views):
        col.image(im, width='stretch')
        col.markdown(f'<div class="caption">{cap}</div>', unsafe_allow_html=True)


def segment(unet, img):
    return unet.predict(np.asarray(img.resize((256, 256)), dtype=np.float32)[None] / 255.0, verbose=0)[0, :, :, 0] > 0.5


clf, unet = load_models()
cls_m, seg_m = load_metrics('metrics.json'), load_metrics('segmentation_metrics.json')

with st.sidebar:
    st.header('⚙️ Settings')
    threshold = st.slider('Decision threshold', 0.05, 0.95, 0.5, 0.05,
                          help='Probability above which a scan is flagged as tumor.')
    alpha = st.slider('Overlay opacity', 0.0, 0.85, 0.45, 0.05)
    st.divider()
    st.markdown('**Models**')
    st.markdown(f"{'✅' if clf else '❌'} ResNet50 classifier  \n{'✅' if unet else '❌'} U-Net segmenter")
    st.divider()
    st.caption('⚠️ Research demo only. Not a medical device and not for diagnosis.')

tab_detect, tab_seg, tab_perf = st.tabs(['🔍 Tumor detection', '🩻 Segmentation', '📊 Model performance'])

# ---------------------------------------------------------------- detection
with tab_detect:
    if clf is None:
        st.error('No trained classifier found. Run `python -m scripts.train_classifier` first.')
    else:
        src = st.radio('Image source', ['Upload', 'Sample from dataset'], horizontal=True, key='det_src',
                       label_visibility='collapsed') if os.path.isdir(SAMPLES) else 'Upload'
        img = None
        if src == 'Upload':
            f = st.file_uploader('Upload a brain MRI image', type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])
            if f:
                img = Image.open(f).convert('RGB')
        else:
            c1, c2 = st.columns([1, 3])
            folder = c1.selectbox('Class', ['Brain Tumor', 'Healthy'])
            files = sorted(os.listdir(os.path.join(SAMPLES, folder)))
            name = c2.select_slider('Scan', files, value=files[0])
            img = Image.open(os.path.join(SAMPLES, folder, name)).convert('RGB')

        if img is None:
            st.info('👆 Upload an MRI slice (JPG, PNG or TIFF) to get started.')
        else:
            img224 = img.resize((224, 224))
            with st.spinner('Analyzing scan...'):
                heat, prob = grad_cam(clf, np.asarray(img224, dtype=np.float32)[None])
            is_tumor = prob >= threshold
            left, right = st.columns([1, 2], gap='large')
            with left:
                st.markdown(f"""
                <div class="verdict {'tumor' if is_tumor else 'healthy'}">
                  <div class="label">Prediction</div>
                  <div class="value">{'🔴 Tumor detected' if is_tumor else '🟢 No tumor detected'}</div>
                </div>""", unsafe_allow_html=True)
                st.metric('Tumor probability', f'{prob:.1%}')
                st.progress(prob)
                st.metric('Confidence in prediction', f'{(prob if is_tumor else 1 - prob):.1%}')
            with right:
                views = [(img224, 'Input scan'), (overlay(img224, heat, alpha), 'Grad-CAM: what drove the "tumor" score')]
                if unet is not None:
                    views.append((overlay(img224, segment(unet, img).astype(float), alpha + 0.15, 'autumn'),
                                  'U-Net tumor mask (rough guide)'))
                show_images(views)

# ---------------------------------------------------------------- segmentation
with tab_seg:
    if unet is None:
        st.error('No trained U-Net found. Run `python -m scripts.train_unet` first.')
    elif not os.path.isdir(LGG):
        st.info('Download the LGG dataset into `data/lgg/kaggle_3m` to browse held-out patients (see README).')
    else:
        st.markdown('Browse **held-out test patients** from the LGG dataset. The U-Net never saw them during training. '
                    'Compare its mask with the expert-drawn ground truth.')
        test = lgg_test_patients()
        c1, c2 = st.columns([1, 3])
        patient = c1.selectbox('Test patient', list(test), format_func=lambda p: p.replace('TCGA_', ''))
        files = test[patient]
        # open on the slice with the largest tumor so the first view is informative
        sizes = [np.asarray(Image.open(os.path.join(LGG, patient, f[:-4] + '_mask.tif'))).sum() for f in files]
        name = c2.select_slider('Slice', files, value=files[int(np.argmax(sizes))],
                                format_func=lambda f: f.rsplit('_', 1)[1][:-4])
        img = Image.open(os.path.join(LGG, patient, name)).convert('RGB').resize((256, 256))
        truth = np.asarray(Image.open(os.path.join(LGG, patient, name[:-4] + '_mask.tif')).convert('L')
                           .resize((256, 256), Image.NEAREST)) > 127
        pred = segment(unet, img)
        inter, total = (truth & pred).sum(), truth.sum() + pred.sum()

        left, right = st.columns([1, 3], gap='large')
        with left:
            st.metric('Slice Dice', f'{2 * inter / total:.2f}' if total else '1.00 (both empty)')
            st.metric('Tumor area: expert', f'{truth.mean():.1%} of slice')
            st.metric('Tumor area: U-Net', f'{pred.mean():.1%} of slice')
        with right:
            show_images([(img, 'MRI (T1 / FLAIR / T1-Gd as RGB)'),
                         (overlay(img, truth.astype(float), alpha + 0.15, 'autumn'), 'Ground truth (expert)'),
                         (overlay(img, pred.astype(float), alpha + 0.15, 'autumn'), 'U-Net prediction')])

# ---------------------------------------------------------------- performance
with tab_perf:
    if cls_m:
        st.subheader('Classifier · ResNet50')
        st.caption(f"{cls_m['split_sizes']['test']} held-out images. Near-duplicate scans are never split across train and test.")
        for col, (k, label, fmt) in zip(st.columns(5), [
                ('accuracy', 'Accuracy', '{:.1%}'), ('roc_auc', 'ROC AUC', '{:.3f}'),
                ('recall_tumor', 'Sensitivity', '{:.1%}'), ('specificity', 'Specificity', '{:.1%}'),
                ('f1_tumor', 'F1 (tumor)', '{:.3f}')]):
            col.metric(label, fmt.format(cls_m[k]))
        c1, c2, c3 = st.columns([1, 1, 2.2])
        c1.image(os.path.join(RESULTS, 'confusion_matrix.png'), width='stretch')
        c2.image(os.path.join(RESULTS, 'roc_curve.png'), width='stretch')
        c3.image(os.path.join(RESULTS, 'training_curves.png'), width='stretch')
        st.image(os.path.join(RESULTS, 'gradcam_examples.png'), width='stretch')
    if seg_m:
        st.divider()
        st.subheader('Segmenter · U-Net')
        st.caption(f"{seg_m['split_sizes']['test']} slices from held-out patients (patient-level split).")
        for col, (k, label, fmt) in zip(st.columns(5), [
                ('dice_global', 'Dice', '{:.3f}'), ('iou_global', 'IoU', '{:.3f}'),
                ('dice_tumor_slices', 'Dice / tumor slice', '{:.3f}'),
                ('tumor_slice_detection_rate', 'Tumor slices found', '{:.1%}'),
                ('empty_slice_false_positive_rate', 'False-positive slices', '{:.1%}')]):
            col.metric(label, fmt.format(seg_m[k]))
        st.image(os.path.join(RESULTS, 'segmentation_examples.png'), width='stretch')
    if not (cls_m or seg_m):
        st.info('No results yet. Train the models to populate this page.')
