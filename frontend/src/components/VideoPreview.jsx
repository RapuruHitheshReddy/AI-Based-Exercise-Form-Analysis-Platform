import React from "react";

export default function VideoPreview({ videos }) {
  if (!videos) return null;

  const entries = Object.entries(videos);

  return (
    <div className="mt-10">
      <h2 className="text-2xl font-bold text-center text-blue-700 mb-6">
        🎬 Processed Videos
      </h2>

      <div className="grid sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
        {entries.map(([key, relPath]) => {
          const label =
            key
              .replace(/_/g, " ")
              .replace(/\b\w/g, (l) => l.toUpperCase()) || "Video";

          const videoUrl = relPath.startsWith("/")
            ? `http://localhost:5000${relPath}`
            : `http://localhost:5000/${relPath}`;

          return (
            <div
              key={key}
              className="bg-white rounded-xl shadow-md border border-gray-200 p-3 text-center"
            >
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                {label}
              </h3>

              <video
                src={videoUrl}
                width="100%"
                height="240"
                controls
                preload="metadata"
                className="w-full rounded-lg border border-gray-300"
                onError={(e) =>
                  console.error("🎞️ Video failed:", e.target.src)
                }
              >
                Your browser does not support the video tag.
              </video>

              <p className="text-sm text-gray-600 mt-2">
                {key.includes("annotated")
                  ? "AI-Annotated Feedback Video"
                  : key.includes("stick")
                  ? "Stick Figure Visualization"
                  : "Original Processed Clip"}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
