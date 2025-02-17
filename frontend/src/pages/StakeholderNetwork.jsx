import { Search, Filter, UserPlus, Mail, Phone } from 'lucide-react';
import { useState } from 'react';

const StakeholderNetwork = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredStakeholders, setFilteredStakeholders] = useState(stakeholders);

  const handleSearch = (e) => {
    const term = e.target.value.toLowerCase();
    setSearchTerm(term);
    
    const filtered = stakeholders.filter(stakeholder => 
      stakeholder.name.toLowerCase().includes(term) ||
      stakeholder.role.toLowerCase().includes(term) ||
      stakeholder.organization.toLowerCase().includes(term) ||
      stakeholder.tags.some(tag => tag.toLowerCase().includes(term))
    );
    
    setFilteredStakeholders(filtered);
  };

  return (
    <div className="space-y-8">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Stakeholder Network</h1>
          <p className="text-gray-600 mt-2">Manage and analyze key relationships</p>
        </div>
        <button className="btn-primary flex items-center gap-2">
          <UserPlus size={20} />
          Add Stakeholder
        </button>
      </header>

      <div className="grid md:grid-cols-3 gap-6">
        <div className="md:col-span-2 card">
          <div className="flex gap-4 mb-6">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
              <input
                type="text"
                placeholder="Search stakeholders..."
                className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20"
                value={searchTerm}
                onChange={handleSearch}
              />
            </div>
            <button className="flex items-center gap-2 px-4 py-2 border border-gray-200 rounded-lg hover:bg-gray-50">
              <Filter size={20} />
              Filters
            </button>
          </div>

          <div className="space-y-4">
            {filteredStakeholders.map((stakeholder, index) => (
              <StakeholderCard key={index} {...stakeholder} />
            ))}
          </div>
        </div>

        <div className="space-y-6">
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Network Statistics</h2>
            <div className="space-y-4">
              <StatItem label="Total Stakeholders" value={stakeholders.length.toString()} />
              <StatItem label="Active Relationships" value="89" />
              <StatItem label="Influence Score" value="7.8/10" />
              <StatItem label="Recent Interactions" value="24" />
            </div>
          </div>

          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
            <div className="space-y-2">
              <button className="w-full text-left px-4 py-2 hover:bg-gray-50 rounded-lg flex items-center gap-2">
                <Mail size={20} className="text-gray-500" />
                Send Email
              </button>
              <button className="w-full text-left px-4 py-2 hover:bg-gray-50 rounded-lg flex items-center gap-2">
                <Phone size={20} className="text-gray-500" />
                Schedule Call
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const StakeholderCard = ({ name, role, organization, influence, tags, lastContact }) => (
  <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
    <div className="flex items-start justify-between mb-2">
      <div>
        <h3 className="font-semibold text-lg">{name}</h3>
        <p className="text-gray-600">{role} at {organization}</p>
      </div>
      <div className="text-right">
        <p className="text-sm text-gray-600">Influence Score</p>
        <p className="font-semibold text-primary">{influence}/10</p>
      </div>
    </div>

    <div className="flex items-center justify-between mt-4">
      <div className="flex gap-2">
        {tags.map((tag, index) => (
          <span key={index} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-sm">
            {tag}
          </span>
        ))}
      </div>
      <p className="text-sm text-gray-500">Last Contact: {lastContact}</p>
    </div>
  </div>
);

const StatItem = ({ label, value }) => (
  <div className="flex justify-between items-center">
    <span className="text-gray-600">{label}</span>
    <span className="font-semibold">{value}</span>
  </div>
);

const stakeholders = [
  {
    name: "Sarah Johnson",
    role: "Committee Chair",
    organization: "House Energy Committee",
    influence: 8.5,
    tags: ["Energy", "Climate"],
    lastContact: "2 days ago"
  },
  {
    name: "Michael Chen",
    role: "Policy Director",
    organization: "Tech Innovation Council",
    influence: 7.2,
    tags: ["Technology", "Innovation"],
    lastContact: "1 week ago"
  },
  {
    name: "Amanda Rodriguez",
    role: "Senior Advisor",
    organization: "Department of Commerce",
    influence: 8.9,
    tags: ["Trade", "Economics"],
    lastContact: "3 days ago"
  }
];

export default StakeholderNetwork;