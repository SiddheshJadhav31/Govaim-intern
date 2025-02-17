import { useNavigate } from 'react-router-dom';
import { ArrowRight, Shield, Zap, Network, ChevronDown } from 'lucide-react';
import { useEffect } from 'react';
import AOS from 'aos';
import 'aos/dist/aos.css';

const Landing = () => {
  const navigate = useNavigate();

  useEffect(() => {
    AOS.init({
      duration: 1000,
      once: true,
      easing: 'ease-in-out'
    });
  }, []);

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="relative bg-gradient-to-br from-primary to-secondary text-white min-h-screen flex items-center">
        <div className="absolute inset-0 bg-black opacity-10"></div>
        <div className="container mx-auto px-6 py-24 relative">
          <div className="max-w-3xl space-y-8" data-aos="fade-up">
            <h1 className="text-6xl font-bold mb-6 animate-fade-in">
              Transform Your{' '}
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-accent to-accent-light">
                Legislative Strategy
              </span>{' '}
              with AI
            </h1>
            <p className="text-xl mb-8 text-gray-100 leading-relaxed" data-aos="fade-up" data-aos-delay="200">
              GovAim empowers organizations with AI-driven insights for more effective legislative advocacy and stakeholder engagement. Experience the future of policy management.
            </p>
            <div className="flex gap-4" data-aos="fade-up" data-aos-delay="400">
              <button 
                onClick={() => navigate('/app')}
                className="bg-accent hover:bg-accent-light text-primary font-semibold px-8 py-3 rounded-lg flex items-center gap-2 transform hover:scale-105 transition-all duration-300"
              >
                Get Started
                <ArrowRight size={20} />
              </button>
              <button 
                onClick={() => navigate('/auth/login')}
                className="border-2 border-white hover:bg-white/10 font-semibold px-8 py-3 rounded-lg flex items-center gap-2 transition-all duration-300"
              >
                Learn More
              </button>
            </div>
          </div>
          <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 animate-bounce">
            <ChevronDown size={32} className="text-white/80" />
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="container mx-auto px-6 py-24 bg-gray-50">
        <h2 className="text-4xl font-bold text-center mb-16" data-aos="fade-up">
          Powerful Features for Modern Governance
        </h2>
        <div className="grid md:grid-cols-3 gap-12">
          <FeatureCard 
            icon={<Zap className="text-accent" size={32} />}
            title="Predictive Analytics"
            description="Leverage AI to predict legislative outcomes and identify key opportunities for engagement."
            delay={0}
          />
          <FeatureCard 
            icon={<Network className="text-accent" size={32} />}
            title="Stakeholder Mapping"
            description="Visualize and analyze complex relationships between legislators, organizations, and key stakeholders."
            delay={200}
          />
          <FeatureCard 
            icon={<Shield className="text-accent" size={32} />}
            title="Compliance Tracking"
            description="Stay compliant with automated monitoring and real-time alerts on policy changes."
            delay={400}
          />
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-gradient-to-r from-primary-dark to-primary py-20 text-white">
        <div className="container mx-auto px-6">
          <div className="grid md:grid-cols-3 gap-12 text-center">
            <StatCard number="95%" text="Accuracy Rate" delay={0} />
            <StatCard number="500+" text="Active Users" delay={200} />
            <StatCard number="24/7" text="Support Available" delay={400} />
          </div>
        </div>
      </div>
    </div>
  );
};

const FeatureCard = ({ icon, title, description, delay }) => (
  <div 
    className="card hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2"
    data-aos="fade-up"
    data-aos-delay={delay}
  >
    <div className="mb-4">{icon}</div>
    <h3 className="text-xl font-semibold mb-2">{title}</h3>
    <p className="text-gray-600">{description}</p>
  </div>
);

const StatCard = ({ number, text, delay }) => (
  <div 
    className="space-y-2"
    data-aos="zoom-in"
    data-aos-delay={delay}
  >
    <div className="text-4xl font-bold">{number}</div>
    <div className="text-lg text-gray-200">{text}</div>
  </div>
);

export default Landing;