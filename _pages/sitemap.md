---
layout: archive
title: "Sitemap"
permalink: /sitemap/
author_profile: true
sitemap: false
---

{% comment %} Pages {% endcomment %}
{% assign pages = site.pages
  | where_exp: "p", "p.title"
  | where_exp: "p", "p.sitemap != false"
  | where_exp: "p", "p.permalink != '/404.html'"
%}

### Pages
{% for p in pages %}
- [{{ p.title }}]({{ p.url | relative_url }})
{% endfor %}

{% for coll in site.collections %}
  {% assign docs = coll.docs
    | where_exp: "d", "d.sitemap != false"
    | where_exp: "d", "d.published != false"
  %}
  {% if docs.size > 0 %}
### {{ coll.label }}
{% for d in docs %}
- [{{ d.title }}]({{ d.url | relative_url }})
{% endfor %}
  {% endif %}
{% endfor %}