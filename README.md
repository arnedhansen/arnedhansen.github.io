# Arne Hansen – Personal Academic Website

A clean, minimal multi-page academic website inspired by radimurban.com, built with Montserrat font, light warm-gray background, and a forked timeline for concurrent roles.

## File structure

```
/
├── index.html          ← Landing page (start here)
├── about.html          ← Bio, skills, memberships
├── experience.html     ← Education + experience timeline (with forks)
├── publications.html   ← Publications & science outreach
├── presentations.html  ← Conference posters & training
├── teaching.html       ← Teaching, supervision, grants & awards
├── style.css           ← Shared stylesheet (all pages use this)
└── assets/
    ├── CV_Arne_COMPLETE.pdf   ← Your CV (linked from index)
    └── photo.jpg              ← Your profile photo (see below)
```

## Setup for GitHub Pages

1. Push all files to your GitHub repository.
2. Go to **Settings → Pages** and set the source to your main branch (`/root` or `/docs` depending on your setup).
3. Your site will be live at `https://arnedhansen.github.io/` (or similar).

## Adding your profile photo

1. Place your photo file in the `assets/` folder, e.g. `assets/photo.jpg`.
2. Open `index.html`, find the `<div class="profile-photo">` block, and:
   - **Uncomment** the `<img>` tag
   - **Remove** the placeholder text `Your photo`

```html
<!-- Before -->
<!-- <img src="assets/photo.jpg" alt="Arne Hansen" /> -->
Your photo

<!-- After -->
<img src="assets/photo.jpg" alt="Arne Hansen" />
```

## Understanding the forked timeline (experience.html)

Roles that started in the **same calendar month** are grouped together inside a
`tl-fork-items` block, marked visually with a left bracket. All other roles are shown
as individual timeline entries in reverse-chronological order.

The fork visual communicates simultaneity clearly without requiring a complex Gantt chart.
The dates on each individual item tell the full story of overlaps.

## Customisation tips

- **Accent color** (links, active nav): Change `--accent: #8b2020;` in `style.css`
- **Background**: Change `--bg: #f0efeb;`
- **Font**: Replace `Montserrat` with any Google Font in both the `<link>` tag and `font-family` in `style.css`
- **CV path**: The CV download links point to `assets/CV_Arne_COMPLETE.pdf` — update if your filename differs
