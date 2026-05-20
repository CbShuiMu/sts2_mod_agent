'use strict';

/**
 * Shared language management for pages that don't use app.js. Also wires the
 * topbar settings button so it navigates to the standalone /settings page
 * from every page consistently.
 */
(function () {
  const STORAGE_KEY = 'sts2-rag-agent-state-v3';
  const i18n = window.STS2_I18N;

  function getLang() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      return saved.lang === 'en' ? 'en' : 'zh';
    } catch {
      return 'zh';
    }
  }

  function setLang(lang) {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      saved.lang = lang;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
    } catch {}
  }

  function t(key, params = {}) {
    return i18n.t(getLang(), key, params);
  }

  function applyTranslations() {
    const lang = getLang();
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';

    document.querySelectorAll('[data-i18n]').forEach(el => {
      el.textContent = i18n.t(lang, el.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      el.setAttribute('placeholder', i18n.t(lang, el.dataset.i18nPlaceholder));
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
      el.setAttribute('title', i18n.t(lang, el.dataset.i18nTitle));
    });
    document.querySelectorAll('[data-i18n-aria]').forEach(el => {
      el.setAttribute('aria-label', i18n.t(lang, el.dataset.i18nAria));
    });

    const btn = document.getElementById('langToggleBtn');
    if (btn) btn.textContent = lang === 'zh' ? 'EN' : '中';
  }

  function toggleLang() {
    const newLang = getLang() === 'zh' ? 'en' : 'zh';
    setLang(newLang);
    applyTranslations();
    window.dispatchEvent(new CustomEvent('sts2langchange', { detail: { lang: newLang } }));
  }

  const SOCIAL_LINKS = [
    {
      href: 'https://discord.gg/dExJy6y5fq',
      label: 'Open Discord community',
      title: 'Discord',
      icon: '/assets/social/discord.png',
    },
    {
      href: 'https://qm.qq.com/q/MlTuoJJkQM',
      label: 'Open QQ group',
      title: 'QQ',
      icon: '/assets/social/qq.png',
    },
  ];

  function injectSocialLinks() {
    const brand = document.querySelector('.site-nav .brand');
    if (!brand || document.querySelector('.site-nav .social-links')) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'site-nav-left';
    brand.parentNode.insertBefore(wrapper, brand);
    wrapper.appendChild(brand);

    const links = document.createElement('div');
    links.className = 'social-links';
    links.setAttribute('aria-label', 'Community links');

    SOCIAL_LINKS.forEach((entry) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'social-link-btn';
      btn.setAttribute('aria-label', entry.label);
      btn.title = entry.title;
      btn.addEventListener('click', () => {
        window.open(entry.href, '_blank', 'noopener,noreferrer');
      });

      const img = document.createElement('img');
      img.src = entry.icon;
      img.alt = '';
      img.setAttribute('aria-hidden', 'true');
      btn.appendChild(img);

      links.appendChild(btn);
    });

    wrapper.appendChild(links);
  }

  function init() {
    injectSocialLinks();

    const btn = document.getElementById('langToggleBtn');
    if (btn) btn.addEventListener('click', toggleLang);

    const settingsBtn = document.getElementById('settingsToggleBtn');
    if (settingsBtn) {
      settingsBtn.addEventListener('click', () => {
        if (window.location.pathname !== '/settings') {
          window.location.href = '/settings';
        }
      });
    }

    applyTranslations();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.STS2_SHARED = { getLang, t, applyTranslations };
})();
