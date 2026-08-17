# Changelog

Version history for `pyfli`, and currently open issues pulled live from
GitHub. Version numbers and dates below follow the releases published on
[PyPI](https://pypi.org/project/pyfli-lib/).

## Open issues

```{raw} html
<div id="gh-issues-widget">
  <p><em>Loading open issues from GitHub…</em></p>
</div>
<script>
(function () {
  var container = document.getElementById("gh-issues-widget");
  fetch("https://api.github.com/repos/vkp217/pyfli-pkg/issues?state=open&per_page=20")
    .then(function (r) {
      if (!r.ok) { throw new Error("GitHub API error: " + r.status); }
      return r.json();
    })
    .then(function (items) {
      var issues = items.filter(function (i) { return !i.pull_request; });
      container.innerHTML = "";
      if (issues.length === 0) {
        var p = document.createElement("p");
        p.textContent = "No open issues right now.";
        container.appendChild(p);
        return;
      }
      var list = document.createElement("ul");
      issues.forEach(function (issue) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.href = issue.html_url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = "#" + issue.number + " " + issue.title;
        li.appendChild(a);
        list.appendChild(li);
      });
      container.appendChild(list);
    })
    .catch(function (err) {
      container.innerHTML = "";
      var p = document.createElement("p");
      p.textContent = "Could not load issues from GitHub (" + err.message + "). ";
      var a = document.createElement("a");
      a.href = "https://github.com/vkp217/pyfli-pkg/issues";
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = "View issues on GitHub";
      p.appendChild(a);
      container.appendChild(p);
    });
})();
</script>
```

[View all issues on GitHub](https://github.com/vkp217/pyfli-pkg/issues)

## Release history

### 0.1.19 — unreleased

*(not yet published to PyPI)*

```{card}
:class-card: sd-bg-info sd-text-white sd-font-weight-bold
:text-align: center

⚡ A major refactor in the library
```

```{card}
:class-card: sd-bg-success sd-text-white

**Features added:**
- *(add notes for this release)*
```

```{card}
:class-card: sd-bg-warning sd-text-dark

**Bugs Fixed:**
- *(add notes for this release)*
```

### [0.1.18](https://pypi.org/project/pyfli-lib/0.1.18/) — 2026-06-25

```{card}
:class-card: sd-bg-success sd-text-white

**Features added:**
- *(add notes for this release)*
```

```{card}
:class-card: sd-bg-warning sd-text-dark

**Bugs Fixed:**
- *(add notes for this release)*
```

### [0.1.17](https://pypi.org/project/pyfli-lib/0.1.17/) — 2026-06-03

```{card}
:class-card: sd-bg-success sd-text-white

**Features added:**
- Simulator
- Detailed Phasor Method
- NLSF, MLE, and plotting methods
```

```{card}
:class-card: sd-bg-warning sd-text-dark

**Bugs Fixed:**
- Subplotting placement error
```
