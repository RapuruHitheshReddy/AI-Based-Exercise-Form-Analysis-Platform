import React from "react";

export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center justify-center py-10 animate-fadeInUp">
      {/* Outer glowing ring */}
      <div className="relative w-20 h-20">
        {/* Spinning gradient ring */}
        <div className="absolute inset-0 border-4 border-transparent border-t-blue-500 border-l-blue-400 rounded-full animate-spin-slow shadow-[0_0_25px_rgba(59,130,246,0.5)]"></div>

        {/* Inner pulsing core */}
        <div className="absolute inset-2 bg-gradient-to-r from-blue-400 to-blue-600 rounded-full animate-pulse shadow-[0_0_15px_rgba(59,130,246,0.6)]"></div>
      </div>

      {/* Text below spinner */}
      <div className="mt-6 text-center">
        <p className="text-lg font-semibold text-blue-700 tracking-wide animate-pulse-slow">
          🔍 Analyzing your movement...
        </p>
        <p className="text-sm text-gray-500 italic mt-1">
          Please wait while AI evaluates your form.
        </p>
      </div>
    </div>
  );
}
