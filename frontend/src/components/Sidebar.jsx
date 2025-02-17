import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Gavel, Network, Settings, Menu } from 'lucide-react';
import { useState } from 'react';

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={`bg-[#1A1F2C] text-white transition-all ${collapsed ? 'w-20' : 'w-64'}`}>
      <div className="p-4 border-b border-[#403E43] flex items-center justify-between">
        {!collapsed && <h1 className="text-xl font-bold text-white">GovAim</h1>}
        <button onClick={() => setCollapsed(!collapsed)} className="p-2 hover:bg-[#403E43] rounded-lg">
          <Menu size={20} className="text-white" />
        </button>
      </div>
      
      <nav className="p-4 space-y-2">
        <NavLink to="/app" end className="flex items-center gap-2 px-4 py-2 text-gray-300 hover:bg-[#403E43] hover:text-white rounded-lg transition-all">
          <LayoutDashboard size={20} />
          {!collapsed && <span>Dashboard</span>}
        </NavLink>
        <NavLink to="/app/legislative" className="flex items-center gap-2 px-4 py-2 text-gray-300 hover:bg-[#403E43] hover:text-white rounded-lg transition-all">
          <Gavel size={20} />
          {!collapsed && <span>Legislative Tracker</span>}
        </NavLink>
        <NavLink to="/app/network" className="flex items-center gap-2 px-4 py-2 text-gray-300 hover:bg-[#403E43] hover:text-white rounded-lg transition-all">
          <Network size={20} />
          {!collapsed && <span>Stakeholder Network</span>}
        </NavLink>
        <NavLink to="/app/settings" className="flex items-center gap-2 px-4 py-2 text-gray-300 hover:bg-[#403E43] hover:text-white rounded-lg transition-all">
          <Settings size={20} />
          {!collapsed && <span>Settings</span>}
        </NavLink>
      </nav>
    </div>
  );
};

export default Sidebar;