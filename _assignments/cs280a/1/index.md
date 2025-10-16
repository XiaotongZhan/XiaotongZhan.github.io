---
title: "Project 1: Colorizing the Prokudin-Gorskii Photo Collection"
collection: assignments
course: "CS280A"
type: "Assignment"
date: 2025-09-12
permalink: /assignments/cs280a/1/
comments: true
share: false
author_profile: false
layout: assignments_page
toc: false
---

<head>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox.min.js"></script>
  <script>
  MathJax = {
    tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
    svg: { fontCache: 'global' }
  };
 </script>
 <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head>

<style>
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  text-align: center;
}

.gallery-item {
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}

.gallery-item img {
  width: 100%;
  max-height: 250px;
  object-fit: contain;
}

.bw-grid { 
  display: grid; 
  grid-template-columns: repeat(2, 1fr); 
  gap: 16px; 
  align-items: end;
}
.bw-item { text-align: center; }
.bw-item a img { width: 100%; max-height: 320px; object-fit: contain; }
.bw-caption { margin-top: 6px; font-size: 0.95em; color: #555; }

@media (max-width: 900px) {
  .bw-grid { grid-template-columns: 1fr; }
}

</style>

<h2>Brief Overview</h2>
<p>
The input to this project is a digitized glass plate, which contains three vertically stacked exposures taken through blue, green, and red filters. 
The overall goal is to reconstruct a single aligned RGB image. My implementation proceeds through the following steps:
</p>
<ol>
  <li><b>Channel Splitting</b><br>
    Each glass plate image is divided into three equal vertical parts corresponding to the blue, green, and red channels (from top to bottom).
  </li>
  <li><b>Alignment</b><br>
    Two approaches implemented:
    <ul>
      <li>Single-scale exhaustive search (baseline) for low-resolution images.</li>
      <li>Image pyramid (coarse-to-fine) for high-resolution .tif images.</li>
    </ul>
    Similarity metric: <b>Normalized Cross-Correlation (NCC)</b>.
  </li>
  <li><b>Stacking</b><br>
    Once the green and red channels are aligned, they are stacked with the blue channel to form an RGB image.
  </li>
  <li><b>Output</b><br>
    The reconstructed RGB image is saved as a .jpg file. Alignment displacements are also reported.
  </li>
</ol>

<h2>Experimental Results</h2>

<h3>Gallery</h3>
<div class="gallery-grid">

  <div class="gallery-item">
    <a href="results/emir.jpg" data-lightbox="gallery" data-title="emir">
      <img src="results/emir.jpg">
    </a>
    <p>emir</p>
  </div>

  <div class="gallery-item">
    <a href="results/italil.jpg" data-lightbox="gallery" data-title="italil">
      <img src="results/italil.jpg">
    </a>
    <p>italil</p>
  </div>

  <div class="gallery-item">
    <a href="results/monastery.jpg" data-lightbox="gallery" data-title="monastery">
      <img src="results/monastery.jpg">
    </a>
    <p>monastery</p>
  </div>

  <div class="gallery-item">
    <a href="results/church.jpg" data-lightbox="gallery" data-title="church">
      <img src="results/church.jpg">
    </a>
    <p>church</p>
  </div>

  <div class="gallery-item">
    <a href="results/three_generations.jpg" data-lightbox="gallery" data-title="three_generations">
      <img src="results/three_generations.jpg">
    </a>
    <p>three_generations</p>
  </div>

  <div class="gallery-item">
    <a href="results/lugano.jpg" data-lightbox="gallery" data-title="lugano">
      <img src="results/lugano.jpg">
    </a>
    <p>lugano</p>
  </div>

  <div class="gallery-item">
    <a href="results/melons.jpg" data-lightbox="gallery" data-title="melons">
      <img src="results/melons.jpg">
    </a>
    <p>melons</p>
  </div>

  <div class="gallery-item">
    <a href="results/lastochikino.jpg" data-lightbox="gallery" data-title="lastochikino">
      <img src="results/lastochikino.jpg">
    </a>
    <p>lastochikino</p>
  </div>

  <div class="gallery-item">
    <a href="results/tobolsk.jpg" data-lightbox="gallery" data-title="tobolsk">
      <img src="results/tobolsk.jpg">
    </a>
    <p>tobolsk</p>
  </div>

  <div class="gallery-item">
    <a href="results/icon.jpg" data-lightbox="gallery" data-title="icon">
      <img src="results/icon.jpg">
    </a>
    <p>icon</p>
  </div>

  <div class="gallery-item">
    <a href="results/cathedral.jpg" data-lightbox="gallery" data-title="cathedral">
      <img src="results/cathedral.jpg">
    </a>
    <p>cathedral</p>
  </div>

  <div class="gallery-item">
    <a href="results/siren.jpg" data-lightbox="gallery" data-title="siren">
      <img src="results/siren.jpg">
    </a>
    <p>siren</p>
  </div>

  <div class="gallery-item">
    <a href="results/self_portrait.jpg" data-lightbox="gallery" data-title="self_portrait">
      <img src="results/self_portrait.jpg">
    </a>
    <p>self_portrait</p>
  </div>

  <div class="gallery-item">
    <a href="results/harvesters.jpg" data-lightbox="gallery" data-title="harvesters">
      <img src="results/harvesters.jpg">
    </a>
    <p>harvesters</p>
  </div>

</div>


<h3>Alignment Displacements (relative to Blue channel)</h3>
<table>
  <tr><th>Image</th><th>Green (x, y)</th><th>Red (x, y)</th></tr>
  <tr><td>emir</td><td>(24, 49)</td><td>(40, 107)</td></tr>
  <tr><td>italil</td><td>(22, 38)</td><td>(36, 77)</td></tr>
  <tr><td>monastery</td><td>(2, -3)</td><td>(2, 3)</td></tr>
  <tr><td>church</td><td>(4, 25)</td><td>(-4, 58)</td></tr>
  <tr><td>three_generations</td><td>(13, 53)</td><td>(10, 112)</td></tr>
  <tr><td>lugano</td><td>(-17, 41)</td><td>(-29, 93)</td></tr>
  <tr><td>melons</td><td>(11, 81)</td><td>(14, 177)</td></tr>
  <tr><td>lastochikino</td><td>(-1, -3)</td><td>(-8, 76)</td></tr>
  <tr><td>tobolsk</td><td>(3, 3)</td><td>(3, 7)</td></tr>
  <tr><td>icon</td><td>(17, 41)</td><td>(23, 90)</td></tr>
  <tr><td>cathedral</td><td>(2, 5)</td><td>(3, 12)</td></tr>
  <tr><td>siren</td><td>(-5, 49)</td><td>(-24, 96)</td></tr>
  <tr><td>self_portrait</td><td>(29, 78)</td><td>(37, 175)</td></tr>
  <tr><td>harvesters</td><td>(18, 60)</td><td>(14, 123)</td></tr>
</table>

<hr>

<h3>My Gallery</h3>
<div class="gallery-grid">

  <div class="gallery-item">
    <a href="myresults/934a.jpg" data-lightbox="mygallery" data-title="934a">
      <img src="myresults/934a.jpg">
    </a>
    <p>934a</p>
  </div>

  <div class="gallery-item">
    <a href="myresults/113u.jpg" data-lightbox="mygallery" data-title="113u">
      <img src="myresults/113u.jpg">
    </a>
    <p>113u</p>
  </div>

  <div class="gallery-item">
    <a href="myresults/525u.jpg" data-lightbox="mygallery" data-title="525u">
      <img src="myresults/525u.jpg">
    </a>
    <p>525u</p>
  </div>

  <div class="gallery-item">
    <a href="myresults/18u.jpg" data-lightbox="mygallery" data-title="18u">
      <img src="myresults/18u.jpg">
    </a>
    <p>18u</p>
  </div>

  <div class="gallery-item">
    <a href="myresults/218u.jpg" data-lightbox="mygallery" data-title="218u">
      <img src="myresults/218u.jpg">
    </a>
    <p>218u</p>
  </div>

</div>

<h3>Alignment Displacements (relative to Blue channel)</h3>
<table>
  <tr><th>Image</th><th>Green (x, y)</th><th>Red (x, y)</th></tr>
  <tr><td>934a</td><td>(5, 12)</td><td>(-4, 43)</td></tr>
  <tr><td>113u</td><td>(13, 56)</td><td>(14, 123)</td></tr>
  <tr><td>525u</td><td>(-13, 40)</td><td>(-30, 141)</td></tr>
  <tr><td>18u</td><td>(28, 61)</td><td>(43, 132)</td></tr>
  <tr><td>218u</td><td>(-16, 32)</td><td>(-25, 78)</td></tr>
</table>

<h2>Bells & Whistles — Implementation Notes</h2>

<h3>1) Automatic Cropping</h3>

{% include infocard.html title="Goal" content="Remove the black, white, or colorful “rainbow stripe” borders left after splitting and stacking the glass plates. These borders degrade contrast and color perception if left in the image." %}

<p>
<strong>Method</strong> <br>
I combined two complementary signals to detect borders, then used connected component analysis with safety constraints to crop reliably:
<ol>
  <li>Grayscale thresholding (black/white borders)
    <ul>
      <li>Convert image to grayscale and apply Otsu’s automatic threshold.</li>
      <li>If the background is predominantly white, invert the mask.</li>
      <li>Produces mask_otsu (True = foreground).</li>
    </ul>
  </li>
  <li>Channel disagreement (color borders)
    <ul>
      <li>Compute channel disagreement: 
        $$ \text{diff}(x,y) = \max(R,G,B) - \min(R,G,B) $$
      </li>
      <li>Colored stripes cause large channel disagreement, while valid content is more consistent.</li>
      <li>Threshold: $thr = \text{mean(diff)} \times thr\_factor$. Keep low-disagreement regions → mask_diff.</li>
    </ul>
  </li>
  <li>Mask fusion + largest component
    <ul>
      <li>Final mask: $mask = mask\_{otsu} \land mask\_{diff}$.</li>
      <li>Connected component labeling, then take the largest region’s bounding box.</li>
      <li>Add a small margin for safety.</li>
    </ul>
  </li>
  <li>Safety check
    <ul>
      If the proposed crop box is too small (height or width $< min\_keep \times$ original), skip cropping and return the original image.
    </ul>
  </li>
</ol>
</p>


<strong>Advantages</strong>: 

Fast (vectorized + single connected-component pass), works for both black/white and colored borders, and min_keep prevents “cutting the image in half.”


<strong>Failure cases</strong>:
<ul>
  <li>Strong colorful stripes: e.g., self_portrait, where vivid yellow/pink/blue borders are visually similar to foreground. Algorithm misclassifies them as content.</li>
  <li>Weak contrast borders: e.g., icon, where the border brightness and texture resemble interior regions. Thresholding fails, and light top/bottom borders remain.</li>
</ul>


<div class="bw-grid">
  <div class="bw-item">
    <a href="compare/emir/emir_crop_off.jpg" data-lightbox="crop" data-title="Cropping OFF">
      <img src="compare/emir/emir_crop_off.jpg" alt="Cropping OFF">
    </a>
    <div class="bw-caption">Before — Cropping OFF</div>
  </div>
  <div class="bw-item">
    <a href="compare/emir/emir.jpg" data-lightbox="crop" data-title="Cropping ON">
      <img src="compare/emir/emir.jpg" alt="Cropping ON">
    </a>
    <div class="bw-caption">After — Cropping ON</div>
  </div>
</div>

<hr>

<h3>2) Automatic White Balance</h3>

{% include infocard.html title="Goal" content="Estimate the illuminant color bias and transform the image to simulate a neutral illuminant." %}

<strong>Method</strong>

I choose Shades-of-Gray (Minkowski p-norm).

For each channel $c \in \{R,G,B\}$:

$$
m_c = \left( \mathbb{E}\{ |I_c|^p \} \right)^{1/p}, 
\qquad \text{gain}_c \propto \tfrac{1}{m_c}
$$

<p>
Normalize gains to have mean = 1 (preserve overall brightness). Default $p=6$.
</p>

<strong>Why effective?</strong>

Assumes the average scene color should be gray. This allows estimating illuminant bias without explicitly detecting a white patch. Works well for Prokudin-Gorskii plates.


<div class="bw-grid">
  <div class="bw-item">
    <a href="compare/emir/emir_wb_off.jpg" data-lightbox="wb" data-title="White Balance OFF">
      <img src="compare/emir/emir_wb_off.jpg" alt="White Balance OFF">
    </a>
    <div class="bw-caption">Before — White Balance OFF</div>
  </div>
  <div class="bw-item">
    <a href="compare/emir/emir.jpg" data-lightbox="wb" data-title="White Balance ON">
      <img src="compare/emir/emir.jpg" alt="White Balance ON">
    </a>
    <div class="bw-caption">After — White Balance ON</div>
  </div>
</div>

<hr>

<h3>3) Automatic Contrasting</h3>

{% include infocard.html title="Goal" content="Enhance global contrast and visual clarity, without introducing color shifts." %}

<strong>Method</strong>

<ul>
  <li>Operates in CIE-Lab space ($contrast\_in\_lab=True$):</li>
  <li>Percentile stretch (1–99%): clip extreme pixels, rescale L channel to [0,100].</li>
  <li>Optional CLAHE ($use\_clahe=True$): adaptive local histogram equalization, useful for low-contrast details.</li>
  <li>Convert back to RGB (lab2rgb) and clip to [0,1].</li>
</ul>

<p>
Why Lab: Adjusting only the L channel avoids color distortions from RGB scaling.
</p>

<div class="bw-grid">
  <div class="bw-item">
    <a href="compare/emir/emir_contrast_off.jpg" data-lightbox="contrast" data-title="Contrast OFF">
      <img src="compare/emir/emir_contrast_off.jpg" alt="Contrast OFF">
    </a>
    <div class="bw-caption">Before — Contrast OFF</div>
  </div>
  <div class="bw-item">
    <a href="compare/emir/emir.jpg" data-lightbox="contrast" data-title="Contrast ON">
      <img src="compare/emir/emir.jpg" alt="Contrast ON">
    </a>
    <div class="bw-caption">After — Contrast ON</div>
  </div>
</div>

<hr>

<h3>4) Better Color Mapping</h3>

{% include infocard.html title="Goal" content="The three Prokudin-Gorskii plates were captured through B/G/R filters, but they don’t perfectly correspond to sRGB R/G/B channels. Relying only on white balance often leaves residual color bias. I added a global color correction step to further suppress bias and reduce dependence on white balance accuracy." %}

<strong>Method</strong>

<ul>
  <li>Simplified robust version: Gray-World diagonal matrix
  $$
  M = \mathrm{diag}\!\left(\tfrac{\bar g}{\bar R}, \tfrac{\bar g}{\bar G}, \tfrac{\bar g}{\bar B}\right), 
  \qquad \bar g = \tfrac{1}{3}(\bar R+\bar G+\bar B)
  $$
  This rescales each channel so their means are equal, suppressing residual tints while avoiding artificial colors.
  </li>
  <li>Why not a full $3 \times 3$ least-squares matrix:<br>
  In experiments, full $3 \times 3$ fitting overfits border/edge noise, introduces cross-channel coupling, and sometimes causes saturation or grayscale artifacts. The diagonal Gray-World matrix is more stable for historical plates.
  </li>
</ul>

<div class="bw-grid">
  <div class="bw-item">
    <a href="compare/emir/emir_colormap_off.jpg" data-lightbox="cmap" data-title="Color Mapping OFF">
      <img src="compare/emir/emir_colormap_off.jpg" alt="Color Mapping OFF">
    </a>
    <div class="bw-caption">Before — Color Mapping OFF</div>
  </div>
  <div class="bw-item">
    <a href="compare/emir/emir.jpg" data-lightbox="cmap" data-title="Color Mapping ON">
      <img src="compare/emir/emir.jpg" alt="Color Mapping ON">
    </a>
    <div class="bw-caption">After — Color Mapping ON</div>
  </div>
</div>

<hr>

<h3>5) Better Features for Alignment</h3>

<strong>Method</strong>

Use gradient magnitude instead of raw intensities when scoring alignment.
<ul>
  <li>Compute $$ \text{grad_mag} = \sqrt{(\text{Sobel}_x)^2 + (\text{Sobel}_y)^2} $$</li>
  <li>NCC scores are computed on gradients, which are more invariant to illumination/color differences.</li>
</ul>

<strong>Why effective?</strong>

Helps align channels with strong color bias (e.g., emir), where pixel intensities differ but edges are consistent.

<div class="bw-grid">
  <div class="bw-item">
    <a href="compare/emir/emir_gradient_off.jpg" data-lightbox="grad" data-title="Gradient Feature OFF">
      <img src="compare/emir/emir_gradient_off.jpg" alt="Gradient Feature OFF">
    </a>
    <div class="bw-caption">Before — Gradient Feature OFF</div>
  </div>
  <div class="bw-item">
    <a href="compare/emir/emir.jpg" data-lightbox="grad" data-title="Gradient Feature ON">
      <img src="compare/emir/emir.jpg" alt="Gradient Feature ON">
    </a>
    <div class="bw-caption">After — Gradient Feature ON</div>
  </div>
</div>

<hr>

<h3>6) Coarse-to-Fine Image Pyramid</h3>

<strong>Method</strong>

<ol>
  <li>If the image is small enough (min side ≤ min_size), run single-scale exhaustive search within [-max_shift, max_shift].</li>
  <li>Otherwise, downsample moving/reference by ~0.5, recursively estimate coarse displacement.</li>
  <li>Upscale coarse shift to current resolution, refine with a local search window.</li>
  <li>Return the refined displacement.</li>
</ol>

<p>
Complexity advantage: Reduces quadratic cost of full-resolution search. For large TIFs, runtime improves dramatically (2+ minutes to around 40s).
</p>