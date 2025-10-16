/* Build sidebar table of contents without relying on Liquid plugins. */
(function() {
  var initialized = false;

  function slugify(text, used) {
    var base = (text || '')
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '');
    if (!base) base = 'section';
    var slug = base;
    var i = 2;
    while (used[slug]) {
      slug = base + '-' + i;
      i += 1;
    }
    used[slug] = true;
    return slug;
  }

  function pruneEmptyLists(root) {
    var lists = root.querySelectorAll('ul');
    for (var i = lists.length - 1; i >= 0; i -= 1) {
      var list = lists[i];
      if (!list.children.length && list !== root) {
        list.parentNode.removeChild(list);
      }
    }
  }

  function buildOnce() {
    var content = document.querySelector('.page__content');
    if (!content) return;
    var navs = document.querySelectorAll('.page__toc .toc, .sidebar__toc .toc');
    if (!navs.length) return;

    Array.prototype.forEach.call(navs, function(nav) {
      var minLevel = parseInt(nav.getAttribute('data-h-min'), 10) || 1;
      var maxLevel = parseInt(nav.getAttribute('data-h-max'), 10) || 6;
      var list = nav.querySelector('.toc__menu');
      if (!list) return;

      var headings = content.querySelectorAll('h1,h2,h3,h4,h5,h6');
      var usedIds = {};
      var filtered = [];

      Array.prototype.forEach.call(headings, function(node) {
        var level = parseInt(node.tagName.slice(1), 10);
        if (level >= minLevel && level <= maxLevel) {
          filtered.push(node);
          if (node.id) usedIds[node.id] = true;
        }
      });

      if (!filtered.length) {
        nav.style.display = 'none';
        return;
      }

      list.innerHTML = '';
      var stack = [{ level: minLevel - 1, element: list }];

      filtered.forEach(function(head) {
        var level = parseInt(head.tagName.slice(1), 10);
        var text = (head.textContent || head.innerText || '').trim();
        if (!text) return;

        level = Math.min(Math.max(level, minLevel), maxLevel);

        if (!head.id) {
          head.id = slugify(text, usedIds);
        } else if (usedIds[head.id]) {
          head.id = slugify(head.id, usedIds);
        } else {
          usedIds[head.id] = true;
        }

        while (stack.length > 1 && stack[stack.length - 1].level >= level) {
          stack.pop();
        }
        var parent = stack[stack.length - 1];
        if (level > parent.level + 1) {
          level = parent.level + 1;
        }

        var parentList = parent.element;
        var li = document.createElement('li');
        var link = document.createElement('a');
        link.setAttribute('href', '#' + head.id);
        link.textContent = text;
        li.appendChild(link);
        parentList.appendChild(li);

        var child = document.createElement('ul');
        child.className = 'toc__menu';
        li.appendChild(child);
        stack.push({ level: level, element: child });
      });

      pruneEmptyLists(nav);
      nav.style.display = '';
    });
  }

  function schedule() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', buildOnce, { once: true });
    } else {
      buildOnce();
    }
  }

  schedule();
  document.addEventListener('turbolinks:load', buildOnce);
  document.addEventListener('pjax:complete', buildOnce);
})();
