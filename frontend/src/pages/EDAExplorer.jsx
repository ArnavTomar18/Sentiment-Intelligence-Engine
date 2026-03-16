import React, { useState } from 'react'
import Topbar from '../components/layout/Topbar'
import EDAGallery from '../components/common/EDAGallery'

const DOMAINS = [
{ key: 'hotel', label: '🏨 Hotel', images: [
  { path: '/eda/hotel/rating_distribution.png', caption: 'Rating Distribution' },
  { path: '/eda/hotel/sentiment_split.png', caption: 'Sentiment Split' },
  { path: '/eda/hotel/review_length.png', caption: 'Review Length Distribution' },
  { path: '/eda/hotel/top_keywords.png', caption: 'Top Keywords' },
  { path: '/eda/hotel/length_by_rating.png', caption: 'Review Length by Rating' },
  { path: '/eda/hotel/wordcloud.png', caption: 'Word Cloud' },
]},
  { key: 'news',    label: '📰 News',          images: [
    { path: 'news/label_distribution.png',   caption: 'Fake vs Real Distribution' },
    { path: 'news/subject_distribution.png', caption: 'Subject Distribution' },
    { path: 'news/article_length.png',       caption: 'Article Length Distribution' },
    { path: 'news/wordcloud_fake.png',       caption: 'Fake News Word Cloud' },
    { path: 'news/wordcloud_real.png',       caption: 'Real News Word Cloud' },
  ]},
  { key: 'fashion', label: '👗 Fashion',       images: [
    { path: 'fashion/rating_distribution.png', caption: 'Rating Distribution' },
    { path: 'fashion/top_items.png',           caption: 'Top Items Reviewed' },
    { path: 'fashion/aspect_counts.png',       caption: 'Aspect Mention Count' },
    { path: 'fashion/wordcloud.png',           caption: 'Word Cloud' },
  ]},
  { key: 'app',     label: '📱 App Reviews',   images: [
    { path: 'app/rating_distribution.png',   caption: 'Rating Distribution' },
    { path: 'app/top_apps.png',              caption: 'Top Apps Reviewed' },
    { path: 'app/feedback_distribution.png', caption: 'Feedback Type Distribution' },
    { path: 'app/wordcloud.png',             caption: 'Word Cloud' },
  ]},
  { key: 'ott',     label: '🎬 OTT Content',   images: [
    { path: 'ott/content_type.png',          caption: 'Content Type Distribution' },
    { path: 'ott/platform_distribution.png', caption: 'Platform Distribution' },
    { path: 'ott/top_genres.png',            caption: 'Top Genres' },
    { path: 'ott/release_year_trend.png',    caption: 'Content by Release Year' },
    { path: 'ott/wordcloud.png',             caption: 'Description Word Cloud' },
  ]},
]

export default function EDAExplorer() {
  const [active, setActive] = useState('hotel')
  const domain = DOMAINS.find(d => d.key === active)

  return (
    <>
      <Topbar title="EDA Explorer" subtitle="Exploratory Data Analysis — all 5 datasets" />
      <div className="page-body">
        <div className="page-header fade-up">
          <div className="breadcrumb"><span>SIE</span> / EDA Explorer</div>
          <h1 className="page-title">EDA <span>Explorer</span></h1>
          <p className="page-desc">Exploratory Data Analysis visualisations across all 5 review domains.</p>
        </div>

        {/* Domain selector */}
        <div className="task-tabs fade-up fade-up-1">
          {DOMAINS.map(({ key, label }) => (
            <button key={key} className={`task-tab${active === key ? ' active' : ''}`} onClick={() => setActive(key)}>
              {label}
            </button>
          ))}
        </div>

        <div className="fade-up fade-up-2">
          <div style={{ marginBottom: 20 }}>
            <div className="page-title" style={{ fontSize: 20 }}>
              {domain?.label} — <span style={{ color: 'var(--cyan)' }}>Exploratory Analysis</span>
            </div>
          </div>
          {domain && <EDAGallery images={domain.images} />}
        </div>
      </div>
    </>
  )
}