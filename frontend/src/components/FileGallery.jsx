import React from "react";

export default function FileGallery({ plots, csv }) {
  if (!plots && !csv) return null;

  return (
    <div className="mt-16 animate-fadeInUp">
      <h2 className="text-3xl font-extrabold text-center text-blue-700 mb-10 drop-shadow-sm">
        📈 Analysis Results & Reports
      </h2>

      {/* ---- Plots Section ---- */}
      {plots && (
        <div className="grid sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10 max-w-6xl mx-auto">
          {Object.entries(plots).map(([name, path]) => {
            const cleanName = name
              .replace(/_/g, " ")
              .replace(/\b\w/g, (l) => l.toUpperCase());

            const imgSrc = path.startsWith("/")
              ? `http://localhost:5000${path}`
              : `http://localhost:5000/${path}`;

            return (
              <div
                key={name}
                className="relative bg-white/70 backdrop-blur-md border border-blue-100 rounded-3xl p-4 shadow-lg hover:shadow-2xl hover:-translate-y-2 transition-all duration-300 group"
              >
                {/* Image */}
                <div className="overflow-hidden rounded-2xl">
                  <img
                    src={imgSrc}
                    alt={cleanName}
                    className="w-full h-64 object-contain transition-transform duration-500 group-hover:scale-105"
                    onError={() =>
                      console.warn(`⚠️ Could not load plot: ${imgSrc}`)
                    }
                  />
                </div>

                {/* Title Overlay */}
                <div className="absolute top-4 left-4 bg-blue-600/90 text-white text-sm font-semibold px-4 py-1.5 rounded-full shadow-md backdrop-blur-sm">
                  {cleanName}
                </div>

                {/* Caption */}
                <div className="text-center mt-4">
                  <p className="text-gray-700 font-medium">{cleanName}</p>
                  <p className="text-gray-500 text-sm">
                    AI-generated movement pattern visualization
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ---- CSV Reports ---- */}
      {csv && (
        <div className="mt-12 text-center bg-gradient-to-r from-blue-50 to-blue-100 rounded-3xl p-8 max-w-3xl mx-auto shadow-inner">
          <h3 className="text-2xl font-bold text-blue-800 mb-4">
            📄 CSV Reports
          </h3>

          <p className="text-gray-600 mb-6">
            Download detailed biomechanical and scoring data:
          </p>

          {Object.entries(csv).map(([name, path]) => {
            const csvUrl = path.startsWith("/")
              ? `http://localhost:5000${path}`
              : `http://localhost:5000/${path}`;

            return (
              <a
                key={name}
                href={csvUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-2 mb-3 bg-blue-600 text-white font-medium rounded-full hover:bg-blue-700 transition-all duration-200 shadow-md hover:shadow-lg"
              >
                ⬇️ {name}.csv
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}
