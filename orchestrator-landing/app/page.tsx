"use client"

import { Button } from "@/components/ui/button"
import { ArrowRight, TrendingUp, TrendingDown, BarChart3, Zap, Target, Shield, Menu, X } from "lucide-react"
import { useState } from "react"

export default function HomePage() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div className="min-h-screen bg-white">
      <header className="bg-black text-white px-4 py-4">
        <nav className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-xl font-bold tracking-tight">ORCHESTRATOR</span>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className="hover:text-gray-300 transition-colors">
              Features
            </a>
            <a href="#platform" className="hover:text-gray-300 transition-colors">
              Platform
            </a>
            <a href="#pricing" className="hover:text-gray-300 transition-colors">
              Pricing
            </a>
            <a href="#docs" className="hover:text-gray-300 transition-colors">
              Docs
            </a>
            <Button
              variant="outline"
              size="sm"
              className="border-white text-white hover:bg-white hover:text-black bg-transparent"
            >
              Sign In
            </Button>
          </div>

          <button className="md:hidden" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
            {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </nav>

        {mobileMenuOpen && (
          <div className="md:hidden mt-4 pb-4 border-t border-gray-700">
            <div className="flex flex-col space-y-4 mt-4">
              <a href="#features" className="hover:text-gray-300 transition-colors">
                Features
              </a>
              <a href="#platform" className="hover:text-gray-300 transition-colors">
                Platform
              </a>
              <a href="#pricing" className="hover:text-gray-300 transition-colors">
                Pricing
              </a>
              <a href="#docs" className="hover:text-gray-300 transition-colors">
                Docs
              </a>
              <Button
                variant="outline"
                size="sm"
                className="border-white text-white hover:bg-white hover:text-black w-fit bg-transparent"
              >
                Sign In
              </Button>
            </div>
          </div>
        )}
      </header>

      <main className="flex flex-col items-center justify-center min-h-screen px-4 pt-20">
        <div className="text-center mb-8">
          <div className="flex flex-col items-center mb-6">
            <div className="clean-logo mb-8">
              {/* Bold O with line graph inside */}
              <svg width="160" height="160" viewBox="0 0 160 160" className="mx-auto">
                <defs>
                  <linearGradient id="oGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#000000" />
                    <stop offset="50%" stopColor="#1a1a1a" />
                    <stop offset="100%" stopColor="#333333" />
                  </linearGradient>
                </defs>

                {/* Bold O */}
                <circle cx="80" cy="80" r="60" fill="none" stroke="url(#oGradient)" strokeWidth="12" opacity="0.9" />

                {/* Line graph inside the O */}
                <path
                  d="M30 80 L38 70 L46 90 L54 65 L62 75 L70 60 L80 70 L90 85 L98 75 L106 80 L114 85 L122 75 L130 80"
                  stroke="black"
                  strokeWidth="3"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.8"
                />

                <circle cx="30" cy="80" r="2.5" fill="black" opacity="0.9" />
                <circle cx="38" cy="70" r="2.5" fill="black" opacity="0.9" />
                <circle cx="46" cy="90" r="2.5" fill="black" opacity="0.9" />
                <circle cx="54" cy="65" r="2.5" fill="black" opacity="0.9" />
                <circle cx="62" cy="75" r="2.5" fill="black" opacity="0.9" />
                <circle cx="70" cy="60" r="2.5" fill="black" opacity="0.9" />
                <circle cx="80" cy="70" r="2.5" fill="black" opacity="0.9" />
                <circle cx="90" cy="85" r="2.5" fill="black" opacity="0.9" />
                <circle cx="98" cy="75" r="2.5" fill="black" opacity="0.9" />
                <circle cx="106" cy="80" r="2.5" fill="black" opacity="0.9" />
                <circle cx="114" cy="85" r="2.5" fill="black" opacity="0.9" />
                <circle cx="122" cy="75" r="2.5" fill="black" opacity="0.9" />
                <circle cx="130" cy="80" r="2.5" fill="black" opacity="0.9" />
              </svg>
            </div>

            <h1 className="orchestrator-logo text-6xl md:text-8xl font-bold tracking-tight text-black mb-4">
              ORCHESTRATOR
            </h1>
          </div>

          <div className="relative h-8 overflow-hidden w-full max-w-6xl mb-6">
            <div className="ticker-continuous flex items-center space-x-12 whitespace-nowrap">
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingUp className="h-3 w-3 text-green-500" />
                <span className="text-green-500">AAPL +2.45%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingDown className="h-3 w-3 text-red-500" />
                <span className="text-red-500">TSLA -1.23%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingUp className="h-3 w-3 text-green-500" />
                <span className="text-green-500">BTC +5.67%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingUp className="h-3 w-3 text-green-500" />
                <span className="text-green-500">EUR/USD +0.34%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingDown className="h-3 w-3 text-red-500" />
                <span className="text-red-500">NVDA -0.89%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingUp className="h-3 w-3 text-green-500" />
                <span className="text-green-500">ETH +3.21%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingDown className="h-3 w-3 text-red-500" />
                <span className="text-red-500">GOOGL -0.56%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingUp className="h-3 w-3 text-green-500" />
                <span className="text-green-500">GBP/USD +0.78%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingDown className="h-3 w-3 text-red-500" />
                <span className="text-red-500">MSFT -0.34%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingUp className="h-3 w-3 text-green-500" />
                <span className="text-green-500">DOGE +12.45%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingUp className="h-3 w-3 text-green-500" />
                <span className="text-green-500">JPY/USD +0.89%</span>
              </span>
              <span className="flex items-center space-x-1 text-sm font-medium">
                <TrendingDown className="h-3 w-3 text-red-500" />
                <span className="text-red-500">AMZN -2.11%</span>
              </span>
            </div>
          </div>
        </div>

        <p className="text-xl md:text-2xl text-gray-600 text-center max-w-2xl mb-12 font-light">
          No-code trading platform for equities, crypto, and forex. Trade like a quant/HFT with institutional-grade
          tools.
        </p>


        <div className="flex flex-col sm:flex-row gap-4 mb-20">
          <Button 
            size="lg" 
            className="bg-black text-white hover:bg-gray-800 px-8 py-3 text-lg font-medium"
            onClick={() => {
              if (typeof window !== 'undefined') {
                window.open('../frontend/index.html', '_blank');
              }
            }}
          >
            Start Building
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
          <Button
            variant="outline"
            size="lg"
            className="border-black text-black hover:bg-black hover:text-white px-8 py-3 text-lg font-medium bg-transparent"
          >
            Watch Demo
          </Button>
        </div>

        <div id="features" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-6xl mb-20">
          <div className="text-center p-6 border border-gray-100 rounded-lg hover:border-gray-200 transition-colors">
            <BarChart3 className="h-8 w-8 mx-auto mb-4 text-black" />
            <h3 className="font-semibold text-black mb-2">Visual Strategy Builder</h3>
            <p className="text-sm text-gray-600">
              Drag & drop interface like Scratch. Build complex trading strategies without coding.
            </p>
          </div>
          <div className="text-center p-6 border border-gray-100 rounded-lg hover:border-gray-200 transition-colors">
            <Target className="h-8 w-8 mx-auto mb-4 text-black" />
            <h3 className="font-semibold text-black mb-2">Advanced Indicators</h3>
            <p className="text-sm text-gray-600">SMA, RSI, MACD, Bollinger Bands, and 50+ technical indicators.</p>
          </div>
          <div className="text-center p-6 border border-gray-100 rounded-lg hover:border-gray-200 transition-colors">
            <Zap className="h-8 w-8 mx-auto mb-4 text-black" />
            <h3 className="font-semibold text-black mb-2">24/7 Automation</h3>
            <p className="text-sm text-gray-600">Deploy strategies that execute automatically around the clock.</p>
          </div>
          <div className="text-center p-6 border border-gray-100 rounded-lg hover:border-gray-200 transition-colors">
            <Shield className="h-8 w-8 mx-auto mb-4 text-black" />
            <h3 className="font-semibold text-black mb-2">Comprehensive Backtesting</h3>
            <p className="text-sm text-gray-600">Test strategies with 10+ years of historical market data.</p>
          </div>
        </div>
      </main>

      <section id="platform" className="py-20 px-4 bg-gray-50">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-black mb-8">Built for Professional Traders</h2>
          <p className="text-xl text-gray-600 mb-12 max-w-3xl mx-auto">
            Access the same quantitative tools used by hedge funds and high-frequency trading firms, but with an
            intuitive visual interface that requires no programming knowledge.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white p-8 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold mb-4">Institutional Grade</h3>
              <p className="text-gray-600">
                Real-time market data, sub-millisecond execution, and enterprise-level security.
              </p>
            </div>
            <div className="bg-white p-8 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold mb-4">Multi-Asset Support</h3>
              <p className="text-gray-600">
                Trade stocks, crypto, forex, commodities, and derivatives from one platform.
              </p>
            </div>
            <div className="bg-white p-8 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold mb-4">Risk Management</h3>
              <p className="text-gray-600">Built-in position sizing, stop-losses, and portfolio risk controls.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="pricing" className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-black mb-8">Simple, Transparent Pricing</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="border border-gray-200 rounded-lg p-8">
              <h3 className="text-2xl font-bold mb-4">Starter</h3>
              <p className="text-4xl font-bold mb-6">
                $49<span className="text-lg font-normal text-gray-600">/month</span>
              </p>
              <ul className="text-left space-y-3 mb-8">
                <li>✓ Visual strategy builder</li>
                <li>✓ Basic indicators</li>
                <li>✓ Paper trading</li>
                <li>✓ 1 year backtesting</li>
              </ul>
              <Button className="w-full">Get Started</Button>
            </div>
            <div className="border-2 border-black rounded-lg p-8 relative">
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-black text-white px-4 py-1 rounded-full text-sm">
                Most Popular
              </div>
              <h3 className="text-2xl font-bold mb-4">Professional</h3>
              <p className="text-4xl font-bold mb-6">
                $199<span className="text-lg font-normal text-gray-600">/month</span>
              </p>
              <ul className="text-left space-y-3 mb-8">
                <li>✓ Everything in Starter</li>
                <li>✓ Advanced indicators</li>
                <li>✓ Live trading</li>
                <li>✓ 10 years backtesting</li>
                <li>✓ Portfolio management</li>
              </ul>
              <Button className="w-full bg-black text-white hover:bg-gray-800">Start Free Trial</Button>
            </div>
          </div>
        </div>
      </section>

      <footer className="bg-black text-white py-12 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <span className="text-lg font-bold">ORCHESTRATOR</span>
              </div>
              <p className="text-gray-400 text-sm">Professional trading tools for everyone.</p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>
                  <a href="#" className="hover:text-white transition-colors">
                    Features
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition-colors">
                    Pricing
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition-colors">
                    Documentation
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>
                  <a href="#" className="hover:text-white transition-colors">
                    About
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition-colors">
                    Blog
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition-colors">
                    Careers
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>
                  <a href="#" className="hover:text-white transition-colors">
                    Help Center
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition-colors">
                    Contact
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition-colors">
                    Status
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-sm">&copy; 2024 ORCHESTRATOR. All rights reserved.</p>
            <div className="flex space-x-6 mt-4 md:mt-0">
              <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm">
                Privacy
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm">
                Terms
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm">
                Security
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
