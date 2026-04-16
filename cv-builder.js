const CVBuilder = (() => {
  const parser = new DOMParser();
  const cache = new Map();

  function candidatePaths(path) {
    const normalized = path.replace(/^\.?\//, "");
    const candidates = [normalized];
    if (!normalized.startsWith("/")) {
      candidates.push(`/${normalized}`);
    }
    return [...new Set(candidates)];
  }

  async function fetchDoc(path) {
    if (!cache.has(path)) {
      cache.set(
        path,
        (async () => {
          const attempts = candidatePaths(path);
          let lastError = null;
          for (const candidate of attempts) {
            try {
              const res = await fetch(candidate);
              if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
              }
              const html = await res.text();
              return parser.parseFromString(html, "text/html");
            } catch (error) {
              lastError = error;
            }
          }
          throw new Error(`Failed to load ${path}. Last error: ${lastError ? lastError.message : "unknown"}`);
        })()
      );
    }
    return cache.get(path);
  }

  function headingByText(doc, text) {
    const headings = Array.from(doc.querySelectorAll(".section-heading"));
    return headings.find((node) => node.textContent.trim() === text) || null;
  }

  function collectUntilNextHeading(startNode, itemSelector) {
    const nodes = [];
    let next = startNode ? startNode.nextElementSibling : null;
    while (next && !next.classList.contains("section-heading")) {
      if (next.matches(itemSelector)) {
        nodes.push(next.cloneNode(true));
      }
      next = next.nextElementSibling;
    }
    return nodes;
  }

  function renderInto(target, nodes, limit = null) {
    if (!target) return;
    target.innerHTML = "";
    const count = limit === null ? nodes.length : Math.min(limit, nodes.length);
    for (let i = 0; i < count; i += 1) {
      const node = nodes[i].cloneNode(true);
      if (i === count - 1) {
        node.classList.add("last");
        node.style.borderBottom = "none";
      }
      target.appendChild(node);
    }
  }

  async function loadProfile() {
    const indexDoc = await fetchDoc("index.html");
    const info = indexDoc.querySelector(".profile-info");
    const data = {
      name: "Arne D. Hansen",
      subtitle: "",
      contacts: "",
      links: "",
    };
    if (info) {
      const name = info.querySelector("h1");
      const subtitle = info.querySelector(".profile-subtitle");
      const contacts = info.querySelector(".profile-links");
      const links = info.querySelector(".profile-icon-links");
      if (name) data.name = name.textContent.trim();
      if (subtitle) data.subtitle = subtitle.innerHTML.trim();
      if (contacts) data.contacts = contacts.innerHTML.trim();
      if (links) data.links = links.innerHTML.trim();
    }
    return data;
  }

  async function loadAllSections() {
    const [educationDoc, experienceDoc, publicationsDoc, awardsDoc, skillsDoc, engagementDoc] =
      await Promise.all([
        fetchDoc("education.html"),
        fetchDoc("experience.html"),
        fetchDoc("publications.html"),
        fetchDoc("awards.html"),
        fetchDoc("skills.html"),
        fetchDoc("engagement.html"),
      ]);

    const educationRows = Array.from(educationDoc.querySelectorAll(".timeline .tl-row"));
    const experienceRows = Array.from(experienceDoc.querySelectorAll(".timeline .tl-row"));
    const publicationSections = Array.from(publicationsDoc.querySelectorAll(".pub-section"));
    const outreachIntro = publicationsDoc.querySelector("#science-outreach + p");
    const outreachEntries = Array.from(publicationsDoc.querySelectorAll(".outreach-entry"));
    const awardsEntries = collectUntilNextHeading(
      headingByText(awardsDoc, "Awards"),
      ".award-entry"
    );
    const grantsEntries = collectUntilNextHeading(
      headingByText(awardsDoc, "Grants"),
      ".award-entry"
    );
    const skillsGrid = skillsDoc.querySelector(".skills-grid");
    const workshopsEntries = collectUntilNextHeading(
      headingByText(skillsDoc, "Advanced Training & Workshops"),
      ".pres-entry"
    );
    const institutionalEntries = collectUntilNextHeading(
      headingByText(engagementDoc, "Institutional Activities"),
      ".member-entry"
    );
    const extracurricularEntries = collectUntilNextHeading(
      headingByText(engagementDoc, "Extracurricular Activities"),
      ".member-entry"
    );

    return {
      educationRows,
      experienceRows,
      publicationSections,
      outreachIntro: outreachIntro ? outreachIntro.cloneNode(true) : null,
      outreachEntries,
      awardsEntries,
      grantsEntries,
      skillsGrid: skillsGrid ? skillsGrid.cloneNode(true) : null,
      workshopsEntries,
      institutionalEntries,
      extracurricularEntries,
    };
  }

  return {
    loadProfile,
    loadAllSections,
    renderInto,
  };
})();
