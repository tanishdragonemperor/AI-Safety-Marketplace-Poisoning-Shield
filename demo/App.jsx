import React, { useState, useEffect } from 'react';
import { Shield, Skull, Search, AlertTriangle, CheckCircle, XCircle, Zap, Eye, EyeOff, RefreshCw, ChevronRight, Activity, Target, Lock } from 'lucide-react';

// Simulated data for standalone demo
const generateMockProducts = () => {
  const categories = ['electronics', 'clothing', 'home', 'sports', 'beauty'];
  const attackTypes = ['hidden_characters', 'keyword_stuffing', 'homoglyph', 'fake_reviews', 'metadata_poisoning'];
  
  return Array.from({ length: 20 }, (_, i) => ({
    id: `prod_${i}`,
    title: [
      'Premium Wireless Headphones Pro',
      'Ultra Smart Speaker System',
      'Organic Cotton Classic T-Shirt',
      'Professional Yoga Mat Elite',
      'Natural Face Cream Intensive'
    ][i % 5],
    description: 'High quality product with excellent features and premium materials.',
    category: categories[i % 5],
    price: (Math.random() * 200 + 20).toFixed(2),
    rating: (3 + Math.random() * 2).toFixed(1),
    is_poisoned: i < 6,
    poison_type: i < 6 ? attackTypes[i % 5] : null,
    threat_score: i < 6 ? (0.5 + Math.random() * 0.4).toFixed(2) : (Math.random() * 0.3).toFixed(2)
  }));
};

const ATTACK_INFO = {
  hidden_characters: {
    name: 'Hidden Characters',
    icon: EyeOff,
    color: '#ef4444',
    description: 'Invisible Unicode characters injected to manipulate tokenization and embeddings.',
    example: 'Text\u200blooks\u200bnormal\u200bbut\u200bhas\u200bhidden\u200bchars'
  },
  keyword_stuffing: {
    name: 'Keyword Stuffing',
    icon: Target,
    color: '#f59e0b',
    description: 'Excessive SEO keywords added to artificially boost search rankings.',
    example: 'BEST #1 TOP RATED PREMIUM BESTSELLER AMAZING DEAL'
  },
  homoglyph: {
    name: 'Homoglyph Attack',
    icon: Eye,
    color: '#8b5cf6',
    description: 'Characters replaced with visually identical Unicode lookalikes.',
    example: 'Prеmium (uses Cyrillic е instead of Latin e)'
  },
  fake_reviews: {
    name: 'Fake Reviews',
    icon: AlertTriangle,
    color: '#ec4899',
    description: 'Artificially generated positive reviews to inflate ratings.',
    example: 'AMAZING! Best product EVER! Must buy! 5 stars!!!'
  },
  metadata_poisoning: {
    name: 'Metadata Poisoning',
    icon: Skull,
    color: '#06b6d4',
    description: 'Hidden promotional content and fake metrics in metadata.',
    example: 'Hidden fields with spam: view_count: 999999'
  }
};

export default function MarketplacePoisoningShield() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [products, setProducts] = useState([]);
  const [stats, setStats] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [defenseEnabled, setDefenseEnabled] = useState(false);
  const [searchResults, setSearchResults] = useState({ baseline: [], defended: [] });
  const [selectedAttack, setSelectedAttack] = useState('hidden_characters');
  const [attackDemo, setAttackDemo] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const mockProducts = generateMockProducts();
    setProducts(mockProducts);
    
    const poisoned = mockProducts.filter(p => p.is_poisoned).length;
    setStats({
      total: mockProducts.length,
      poisoned,
      clean: mockProducts.length - poisoned,
      detected: poisoned - 1,
      precision: 0.92,
      recall: 0.88,
      f1: 0.90
    });
  }, []);

  const runAttackDemo = () => {
    setLoading(true);
    setTimeout(() => {
      const info = ATTACK_INFO[selectedAttack];
      setAttackDemo({
        type: selectedAttack,
        original: {
          title: 'Premium Wireless Headphones',
          description: 'High quality audio experience with noise cancellation.',
          rating: 4.2
        },
        poisoned: {
          title: selectedAttack === 'homoglyph' ? 'Prеmium Wirеlеss Hеadphonеs' : 
                 selectedAttack === 'hidden_characters' ? 'Premium\u200b Wireless\u200b Headphones' :
                 selectedAttack === 'keyword_stuffing' ? '[BEST TOP #1] Premium Wireless Headphones' :
                 'Premium Wireless Headphones',
          description: selectedAttack === 'keyword_stuffing' ? 
            'High quality audio experience with noise cancellation. Tags: BEST, TOP RATED, PREMIUM, BESTSELLER, AMAZING' :
            info.example,
          rating: selectedAttack === 'fake_reviews' ? 4.8 : 4.2
        },
        detected: true,
        threatScore: 0.76
      });
      setLoading(false);
    }, 800);
  };

  const runSearch = () => {
    if (!searchQuery) return;
    
    const filtered = products.filter(p => 
      p.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.category.toLowerCase().includes(searchQuery.toLowerCase())
    );
    
    const baseline = [...filtered].sort((a, b) => b.rating - a.rating);
    const defended = [...filtered].sort((a, b) => {
      if (a.is_poisoned && !b.is_poisoned) return 1;
      if (!a.is_poisoned && b.is_poisoned) return -1;
      return b.rating - a.rating;
    });
    
    setSearchResults({ baseline: baseline.slice(0, 5), defended: defended.slice(0, 5) });
  };

  const TabButton = ({ id, label, icon: Icon }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`flex items-center gap-2 px-5 py-3 font-medium transition-all duration-300 border-b-2 ${
        activeTab === id 
          ? 'border-cyan-400 text-cyan-400' 
          : 'border-transparent text-gray-400 hover:text-gray-200'
      }`}
    >
      <Icon size={18} />
      {label}
    </button>
  );

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100" style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}>
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Shield className="w-10 h-10 text-cyan-400" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight">
                  <span className="text-cyan-400">MARKETPLACE</span>
                  <span className="text-gray-400"> POISONING </span>
                  <span className="text-red-400">SHIELD</span>
                </h1>
                <p className="text-xs text-gray-500 tracking-wider">AI SECURITY DEFENSE SYSTEM v1.0</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 rounded-lg border border-gray-700">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-xs text-gray-400">SYSTEM ACTIVE</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b border-gray-800 bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-6 flex gap-1">
          <TabButton id="dashboard" label="Dashboard" icon={Activity} />
          <TabButton id="redteam" label="Red Team" icon={Skull} />
          <TabButton id="blueteam" label="Blue Team" icon={Shield} />
          <TabButton id="search" label="Search Demo" icon={Search} />
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && stats && (
          <div className="space-y-8">
            {/* Stats Grid */}
            <div className="grid grid-cols-4 gap-4">
              {[
                { label: 'Total Products', value: stats.total, color: 'cyan', icon: Activity },
                { label: 'Poisoned', value: stats.poisoned, color: 'red', icon: Skull },
                { label: 'Detected', value: stats.detected, color: 'green', icon: Shield },
                { label: 'F1 Score', value: `${(stats.f1 * 100).toFixed(0)}%`, color: 'purple', icon: Target }
              ].map((stat, i) => (
                <div key={i} className="bg-gray-900/50 border border-gray-800 rounded-xl p-5 hover:border-gray-700 transition-colors">
                  <div className="flex items-center justify-between mb-3">
                    <stat.icon className={`w-5 h-5 text-${stat.color}-400`} style={{ color: stat.color === 'cyan' ? '#22d3ee' : stat.color === 'red' ? '#f87171' : stat.color === 'green' ? '#4ade80' : '#a78bfa' }} />
                    <span className="text-xs text-gray-500 uppercase tracking-wider">{stat.label}</span>
                  </div>
                  <div className="text-3xl font-bold" style={{ color: stat.color === 'cyan' ? '#22d3ee' : stat.color === 'red' ? '#f87171' : stat.color === 'green' ? '#4ade80' : '#a78bfa' }}>
                    {stat.value}
                  </div>
                </div>
              ))}
            </div>

            {/* Attack Distribution */}
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Skull className="w-5 h-5 text-red-400" />
                  Attack Distribution
                </h3>
                <div className="space-y-3">
                  {Object.entries(ATTACK_INFO).map(([key, info]) => {
                    const count = Math.floor(Math.random() * 5) + 1;
                    const Icon = info.icon;
                    return (
                      <div key={key} className="flex items-center gap-3">
                        <Icon size={16} style={{ color: info.color }} />
                        <span className="text-sm text-gray-400 flex-1">{info.name}</span>
                        <div className="w-32 h-2 bg-gray-800 rounded-full overflow-hidden">
                          <div 
                            className="h-full rounded-full transition-all duration-500"
                            style={{ width: `${count * 20}%`, backgroundColor: info.color }}
                          />
                        </div>
                        <span className="text-sm font-mono text-gray-500 w-8">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-cyan-400" />
                  Defense Metrics
                </h3>
                <div className="space-y-4">
                  {[
                    { label: 'Precision', value: stats.precision },
                    { label: 'Recall', value: stats.recall },
                    { label: 'F1 Score', value: stats.f1 }
                  ].map((metric, i) => (
                    <div key={i}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-400">{metric.label}</span>
                        <span className="text-cyan-400 font-mono">{(metric.value * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-full transition-all duration-700"
                          style={{ width: `${metric.value * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Products */}
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">Recent Products Analysis</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-500 border-b border-gray-800">
                      <th className="pb-3 font-medium">Product</th>
                      <th className="pb-3 font-medium">Category</th>
                      <th className="pb-3 font-medium">Status</th>
                      <th className="pb-3 font-medium">Threat Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {products.slice(0, 8).map((product, i) => (
                      <tr key={i} className="hover:bg-gray-800/30 transition-colors">
                        <td className="py-3">
                          <span className="text-gray-200">{product.title}</span>
                        </td>
                        <td className="py-3">
                          <span className="px-2 py-1 bg-gray-800 rounded text-xs text-gray-400">{product.category}</span>
                        </td>
                        <td className="py-3">
                          {product.is_poisoned ? (
                            <span className="flex items-center gap-1 text-red-400">
                              <XCircle size={14} />
                              {ATTACK_INFO[product.poison_type]?.name || 'Poisoned'}
                            </span>
                          ) : (
                            <span className="flex items-center gap-1 text-green-400">
                              <CheckCircle size={14} />
                              Clean
                            </span>
                          )}
                        </td>
                        <td className="py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                              <div 
                                className={`h-full rounded-full ${product.threat_score > 0.5 ? 'bg-red-500' : product.threat_score > 0.3 ? 'bg-yellow-500' : 'bg-green-500'}`}
                                style={{ width: `${product.threat_score * 100}%` }}
                              />
                            </div>
                            <span className="text-xs font-mono text-gray-500">{product.threat_score}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Red Team Tab */}
        {activeTab === 'redteam' && (
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-red-950/50 to-gray-900/50 border border-red-900/50 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <Skull className="w-8 h-8 text-red-400" />
                <div>
                  <h2 className="text-xl font-bold text-red-400">Red Team - Attack Simulator</h2>
                  <p className="text-sm text-gray-400">Simulate various data poisoning attacks on marketplace listings</p>
                </div>
              </div>
            </div>

            {/* Attack Types */}
            <div className="grid grid-cols-5 gap-3">
              {Object.entries(ATTACK_INFO).map(([key, info]) => {
                const Icon = info.icon;
                return (
                  <button
                    key={key}
                    onClick={() => setSelectedAttack(key)}
                    className={`p-4 rounded-xl border transition-all duration-300 text-left ${
                      selectedAttack === key 
                        ? 'border-red-500 bg-red-950/30' 
                        : 'border-gray-800 bg-gray-900/30 hover:border-gray-700'
                    }`}
                  >
                    <Icon size={24} style={{ color: info.color }} className="mb-2" />
                    <div className="text-sm font-medium text-gray-200">{info.name}</div>
                  </button>
                );
              })}
            </div>

            {/* Attack Details */}
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <div className="flex items-start gap-4 mb-6">
                {(() => {
                  const info = ATTACK_INFO[selectedAttack];
                  const Icon = info.icon;
                  return (
                    <>
                      <div className="p-3 rounded-xl" style={{ backgroundColor: `${info.color}20` }}>
                        <Icon size={32} style={{ color: info.color }} />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold mb-1">{info.name}</h3>
                        <p className="text-gray-400 text-sm">{info.description}</p>
                        <div className="mt-3 p-3 bg-gray-800/50 rounded-lg font-mono text-xs text-gray-300 overflow-x-auto">
                          Example: {info.example}
                        </div>
                      </div>
                    </>
                  );
                })()}
              </div>

              <button
                onClick={runAttackDemo}
                disabled={loading}
                className="w-full py-3 bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 rounded-xl font-semibold transition-all duration-300 flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? (
                  <RefreshCw className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    Execute Attack
                  </>
                )}
              </button>
            </div>

            {/* Attack Demo Results */}
            {attackDemo && (
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
                  <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    ORIGINAL (Clean)
                  </h4>
                  <div className="space-y-2">
                    <div className="text-lg font-semibold">{attackDemo.original.title}</div>
                    <div className="text-sm text-gray-400">{attackDemo.original.description}</div>
                    <div className="text-sm">Rating: <span className="text-yellow-400">★ {attackDemo.original.rating}</span></div>
                  </div>
                </div>

                <div className="bg-red-950/20 border border-red-900/50 rounded-xl p-6">
                  <h4 className="text-sm font-medium text-red-400 mb-3 flex items-center gap-2">
                    <XCircle className="w-4 h-4" />
                    POISONED (After Attack)
                  </h4>
                  <div className="space-y-2">
                    <div className="text-lg font-semibold">{attackDemo.poisoned.title}</div>
                    <div className="text-sm text-gray-400">{attackDemo.poisoned.description}</div>
                    <div className="text-sm">Rating: <span className="text-yellow-400">★ {attackDemo.poisoned.rating}</span></div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-red-900/30">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Defense Detection:</span>
                      <span className={attackDemo.detected ? 'text-green-400' : 'text-red-400'}>
                        {attackDemo.detected ? '✓ DETECTED' : '✗ EVADED'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-sm mt-1">
                      <span className="text-gray-400">Threat Score:</span>
                      <span className="text-red-400 font-mono">{attackDemo.threatScore}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Blue Team Tab */}
        {activeTab === 'blueteam' && (
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-cyan-950/50 to-gray-900/50 border border-cyan-900/50 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <Shield className="w-8 h-8 text-cyan-400" />
                <div>
                  <h2 className="text-xl font-bold text-cyan-400">Blue Team - Defense System</h2>
                  <p className="text-sm text-gray-400">Multi-layer defense against marketplace data poisoning</p>
                </div>
              </div>
            </div>

            {/* Defense Layers */}
            <div className="grid grid-cols-3 gap-4">
              {[
                { name: 'Unicode Scanner', desc: 'Detects hidden/zero-width characters', active: true, detections: 12 },
                { name: 'Keyword Analyzer', desc: 'Identifies SEO spam patterns', active: true, detections: 8 },
                { name: 'Homoglyph Detector', desc: 'Finds character substitutions', active: true, detections: 5 },
                { name: 'Review Validator', desc: 'Flags suspicious review patterns', active: true, detections: 15 },
                { name: 'Metadata Inspector', desc: 'Validates metadata integrity', active: true, detections: 7 },
                { name: 'Statistical Analyzer', desc: 'Compares against baseline', active: true, detections: 3 }
              ].map((layer, i) => (
                <div key={i} className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 hover:border-cyan-900/50 transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Lock className="w-4 h-4 text-cyan-400" />
                      <span className="font-medium text-sm">{layer.name}</span>
                    </div>
                    <div className={`w-2 h-2 rounded-full ${layer.active ? 'bg-green-500' : 'bg-gray-600'}`} />
                  </div>
                  <p className="text-xs text-gray-500 mb-3">{layer.desc}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">Detections</span>
                    <span className="text-cyan-400 font-mono text-sm">{layer.detections}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Defense Pipeline Visualization */}
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-6">Defense Pipeline</h3>
              <div className="flex items-center justify-between">
                {['Input', 'Unicode Check', 'Keyword Analysis', 'Homoglyph Scan', 'Review Check', 'Metadata Valid', 'Output'].map((step, i) => (
                  <React.Fragment key={i}>
                    <div className="flex flex-col items-center">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        i === 0 ? 'bg-gray-800' : i === 6 ? 'bg-green-900/50 border border-green-700' : 'bg-cyan-900/30 border border-cyan-800'
                      }`}>
                        {i === 0 ? '📥' : i === 6 ? '✓' : <Shield className="w-5 h-5 text-cyan-400" />}
                      </div>
                      <span className="text-xs text-gray-500 mt-2 text-center max-w-16">{step}</span>
                    </div>
                    {i < 6 && (
                      <ChevronRight className="w-5 h-5 text-gray-600" />
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Search Demo Tab */}
        {activeTab === 'search' && (
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-purple-950/50 to-gray-900/50 border border-purple-900/50 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <Search className="w-8 h-8 text-purple-400" />
                <div>
                  <h2 className="text-xl font-bold text-purple-400">Search Demo - Impact Comparison</h2>
                  <p className="text-sm text-gray-400">See how defense affects search ranking results</p>
                </div>
              </div>
            </div>

            {/* Search Box */}
            <div className="flex gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && runSearch()}
                  placeholder="Search products (try: 'premium', 'headphones', 'electronics')"
                  className="w-full pl-12 pr-4 py-3 bg-gray-900 border border-gray-800 rounded-xl text-gray-200 placeholder-gray-600 focus:border-purple-500 focus:outline-none transition-colors"
                />
              </div>
              <button
                onClick={runSearch}
                className="px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-xl font-medium transition-colors flex items-center gap-2"
              >
                <Search className="w-5 h-5" />
                Search
              </button>
            </div>

            {/* Defense Toggle */}
            <div className="flex items-center gap-4 p-4 bg-gray-900/50 border border-gray-800 rounded-xl">
              <span className="text-gray-400">Defense Mode:</span>
              <button
                onClick={() => setDefenseEnabled(!defenseEnabled)}
                className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                  defenseEnabled 
                    ? 'bg-cyan-600 text-white' 
                    : 'bg-gray-800 text-gray-400'
                }`}
              >
                {defenseEnabled ? 'ENABLED' : 'DISABLED'}
              </button>
              <span className="text-xs text-gray-500">
                {defenseEnabled ? 'Poisoned products are demoted in rankings' : 'Showing raw search results'}
              </span>
            </div>

            {/* Search Results Comparison */}
            {searchResults.baseline.length > 0 && (
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <XCircle className="w-5 h-5 text-red-400" />
                    Without Defense (Baseline)
                  </h3>
                  <div className="space-y-3">
                    {searchResults.baseline.map((product, i) => (
                      <div 
                        key={i} 
                        className={`p-3 rounded-lg border ${
                          product.is_poisoned 
                            ? 'bg-red-950/20 border-red-900/50' 
                            : 'bg-gray-800/30 border-gray-700/50'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-sm">{i + 1}. {product.title}</span>
                          {product.is_poisoned && (
                            <span className="text-xs px-2 py-0.5 bg-red-900/50 text-red-400 rounded">POISONED</span>
                          )}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          Rating: ★ {product.rating} • {product.category}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 pt-4 border-t border-gray-800">
                    <span className="text-sm text-gray-400">
                      Poisoned in top 5: <span className="text-red-400 font-bold">{searchResults.baseline.filter(p => p.is_poisoned).length}</span>
                    </span>
                  </div>
                </div>

                <div className="bg-gray-900/50 border border-cyan-900/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Shield className="w-5 h-5 text-cyan-400" />
                    With Defense (Protected)
                  </h3>
                  <div className="space-y-3">
                    {searchResults.defended.map((product, i) => (
                      <div 
                        key={i} 
                        className={`p-3 rounded-lg border ${
                          product.is_poisoned 
                            ? 'bg-red-950/20 border-red-900/50' 
                            : 'bg-gray-800/30 border-gray-700/50'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-sm">{i + 1}. {product.title}</span>
                          {product.is_poisoned && (
                            <span className="text-xs px-2 py-0.5 bg-red-900/50 text-red-400 rounded">POISONED</span>
                          )}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          Rating: ★ {product.rating} • {product.category}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 pt-4 border-t border-gray-800">
                    <span className="text-sm text-gray-400">
                      Poisoned in top 5: <span className="text-cyan-400 font-bold">{searchResults.defended.filter(p => p.is_poisoned).length}</span>
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-6 text-center text-sm text-gray-500">
          <p>Marketplace Poisoning Shield • AI Security Course Project • Tanish Gupta</p>
        </div>
      </footer>
    </div>
  );
}
