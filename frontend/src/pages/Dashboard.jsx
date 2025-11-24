import React, { useState } from "react";
import UploadForm from "../components/UploadForm";
import FeedbackPanel from "../components/FeedbackPanel";
import VideoPreview from "../components/VideoPreview";
import FileGallery from "../components/FileGallery";

export default function Dashboard() {
  const [result, setResult] = useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-100 via-white to-blue-50 text-gray-800 relative overflow-hidden">
      {/* ✨ Animated Gradient Blobs (background aesthetic) */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute w-72 h-72 bg-blue-300/30 rounded-full blur-3xl top-10 left-20 animate-float"></div>
        <div className="absolute w-96 h-96 bg-indigo-300/20 rounded-full blur-3xl bottom-10 right-10 animate-float-delay"></div>
      </div>

      {/* 🌐 Header */}
      <header className="text-center mb-14 pt-10 animate-fadeIn">
        <h1 className="text-5xl sm:text-6xl font-extrabold text-blue-700 drop-shadow-sm mb-3 tracking-tight">
          🏋️ Form-Learner AI
        </h1>
        <p className="text-lg text-gray-600 font-medium tracking-wide max-w-2xl mx-auto">
          AI-driven posture analysis that helps you perfect your movement — one rep at a time.
        </p>
        <div className="w-24 h-1 bg-blue-500 mx-auto mt-5 rounded-full shadow-md"></div>
      </header>

      {/* 🎥 Upload Form Section */}
      <section className="max-w-3xl mx-auto animate-fadeInUp">
        <UploadForm setResult={setResult} />
      </section>

      {/* 🔹 Divider */}
      <div className="max-w-6xl mx-auto my-12 border-t border-gray-200 opacity-70"></div>

      {/* 📊 Results Section */}
      {result ? (
        <section className="max-w-6xl mx-auto space-y-10 animate-fadeInUp">
          {console.log("📦 Result from backend:", result)}

          {/* ✅ Success Message */}
          {result.message && (
            <div className="text-center">
              <span className="inline-block bg-green-100/80 text-green-800 px-6 py-2.5 rounded-full font-semibold shadow-sm backdrop-blur-sm">
                ✅ {result.message}
              </span>
            </div>
          )}

          {/* 💬 Feedback Section */}
          <FeedbackPanel summary={result} />

          {/* 🎞️ Videos Section */}
          {result.files?.videos && <VideoPreview videos={result.files.videos} />}

          {/* 📈 Graphs & Reports Section */}
          {(result.files?.plots || result.files?.csv) && (
            <FileGallery plots={result.files?.plots} csv={result.files?.csv} />
          )}
        </section>
      ) : (
        <div className="text-center mt-20 text-gray-500 text-lg animate-fadeIn">
          🎥 Upload a video to begin real-time AI analysis and feedback visualization.
        </div>
      )}

      {/* ⚙️ Footer */}
      <footer className="text-center mt-20 py-6 border-t border-gray-200 text-gray-500 text-sm bg-white/40 backdrop-blur-md">
        <p>
          © {new Date().getFullYear()} <span className="font-semibold text-blue-700">Form-Learner AI</span> • Developed for Academic Research
        </p>
      </footer>
    </div>
  );
}
