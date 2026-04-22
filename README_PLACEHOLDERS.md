# Project-Page Placeholders — Items to replace before/after launch

This is a checklist of every spot on `index.html` that is currently a dummy or
temporary value. Update them as you lock down the final paper / media.

## Required updates

- [ ] **arXiv URL** (hero *arXiv* button)
  Currently: `#`
  Replace with the full `https://arxiv.org/abs/XXXX.XXXXX` link once posted.
  Also update the hero button and the `og:url` / meta tags if desired.

- [ ] **Video URL** (hero *Video* button **and** the `<iframe>` in the Video section)
  Currently: hero link `#`, iframe `src="about:blank"`, caption "Video coming soon."
  Replace with the final YouTube embed URL, e.g.
  `https://www.youtube.com/embed/<VIDEO_ID>?rel=0&showinfo=0`. You can then
  remove the "Video coming soon." caption.

- [ ] **BibTeX typo (probable)**
  The BibTeX block currently reads `Hyung Jjn Kim` because that is what was
  provided. If the correct spelling is `Hyungjin Kim` (matches the PDF
  authorship line), fix it in `index.html` and remove this item.

## Optional / nice-to-have

- [ ] **Teaser asset**
  We are reusing the paper's Figure 1 PNG (`static/images/figure1.png`) as the
  teaser. If you later make a short animated teaser (GIF / MP4), swap the
  `<img>` in the `hero teaser` section for a `<video>` element.

- [ ] **Social-card image**
  `og:image` / `twitter:image` point to
  `https://seungyubhan.github.io/sas/static/images/figure1.png`. If you want a
  purpose-made social card (1200×630), drop it at
  `static/images/og.png` and update both meta tags.

- [ ] **Results strip**
  The original Nerfies carousel (selfie videos) was removed. If you want a
  similar strip of qualitative results (e.g. PointGoal1, PointPush1, CarGoal1),
  add videos/GIFs to `static/videos/` and restore a `results-carousel` section
  using Bulma's carousel markup.

## Clean-ups already done

- Google Analytics / gtag snippet removed from `<head>`.
- Nerfies navbar (home link, "More Research" dropdown) removed.
- NeRF-specific sections removed: Results carousel, Visual Effects, Matting,
  Animation / Interpolation slider, Re-rendering, Related Links.
- Unused Nerfies sample assets deleted: entire `static/videos/` and
  `static/interpolation/` folders; `static/images/steve.webm`,
  `interpolate_start.jpg`, `interpolate_end.jpg`.
- Footer's Nerfies attribution kept per CC BY-SA 4.0 license.
- Footer PDF / GitHub icon links retargeted to OpenReview / `seungyubhan/sas`.
