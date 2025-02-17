import { Search, Filter, ArrowUpRight } from 'lucide-react';
import { useState } from 'react';

const LegislativeTracker = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredBills, setFilteredBills] = useState(bills);

  const handleSearch = (e) => {
    const term = e.target.value.toLowerCase();
    setSearchTerm(term);
    
    const filtered = bills.filter(bill => 
      bill.title.toLowerCase().includes(term) ||
      bill.number.toLowerCase().includes(term) ||
      bill.summary.toLowerCase().includes(term) ||
      bill.tags.some(tag => tag.toLowerCase().includes(term))
    );
    
    setFilteredBills(filtered);
  };

  return (
    <div className="space-y-8">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Legislative Tracker</h1>
          <p className="text-gray-600 mt-2">Monitor and analyze legislative activities</p>
        </div>
        <button className="btn-primary">Add Bill</button>
      </header>

      <div className="card">
        <div className="flex gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search bills, committees, or keywords..."
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
          {filteredBills.map((bill, index) => (
            <BillCard key={index} {...bill} />
          ))}
        </div>
      </div>
    </div>
  );
};

const BillCard = ({ title, number, status, probability, summary, tags }) => (
  <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
    <div className="flex items-start justify-between mb-2">
      <div>
        <h3 className="font-semibold text-lg">{title}</h3>
        <p className="text-gray-600">{number}</p>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-right">
          <p className="text-sm text-gray-600">Passage Probability</p>
          <p className="font-semibold text-green-600">{probability}</p>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm ${
          status === 'Active' ? 'bg-green-100 text-green-700' :
          status === 'Pending' ? 'bg-yellow-100 text-yellow-700' :
          'bg-red-100 text-red-700'
        }`}>
          {status}
        </span>
      </div>
    </div>

    <p className="text-gray-600 mb-4">{summary}</p>

    <div className="flex items-center justify-between">
      <div className="flex gap-2">
        {tags.map((tag, index) => (
          <span key={index} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-sm">
            {tag}
          </span>
        ))}
      </div>
      <button className="text-primary hover:text-primary-dark flex items-center gap-1">
        View Details
        <ArrowUpRight size={16} />
      </button>
    </div>
  </div>
);

const bills = [
  {
    title: "Clean Energy Innovation Act",
    number: "H.R. 2468",
    status: "Active",
    probability: "78%",
    summary: "A bill to promote research and development in clean energy technologies and provide tax incentives for clean energy adoption.",
    tags: ["Energy", "Environment", "Tax Policy"]
  },
  {
    title: "Digital Privacy Protection Act",
    number: "S. 982",
    status: "Pending",
    probability: "45%",
    summary: "Comprehensive legislation to enhance consumer privacy protections and establish data handling requirements for technology companies.",
    tags: ["Technology", "Privacy", "Consumer Protection"]
  },
  {
    title: "Infrastructure Modernization Bill",
    number: "H.R. 3571",
    status: "In Committee",
    probability: "62%",
    summary: "Authorization for federal funding to upgrade transportation infrastructure and expand broadband access in rural areas.",
    tags: ["Infrastructure", "Transportation", "Rural Development"]
  }
];

export default LegislativeTracker;