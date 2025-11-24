import React, { useState } from "react";
import { uploadExerciseVideo } from "../api";
import LoadingSpinner from "./LoadingSpinner";

export default function UploadForm({ setResult }) {
  const [exercise, setExercise] = useState("squat");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please upload a video!");

    setLoading(true);
    try {
      const res = await uploadExerciseVideo(exercise, file);
      setResult(res);
    } catch (err) {
      console.error(err);
      alert("Error analyzing video!");
    } finally {
      setLoading(false);
    }
  };

  const exercises = [
    "barbell biceps curl",
    "bench press",
    "chest fly machine",
    "deadlift",
    "decline bench press",
    "hammer curl",
    "hip thrust",
    "incline bench press",
    "lat pulldown",
    "lateral raise",
    "leg extension",
    "leg raises",
    "plank",
    "pull up",
    "push-up",
    "romanian deadlift",
    "russian twist",
    "shoulder press",
    "squat",
    "t bar row",
    "tricep pushdown",
    "tricep dips",
  ];

  return (
    <div className="relative backdrop-blur-xl bg-white/70 border border-white/30 shadow-xl rounded-3xl p-8 max-w-2xl mx-auto transition-transform duration-300 hover:scale-[1.02] hover:shadow-2xl">
      <h2 className="text-3xl font-extrabold text-center text-blue-700 mb-6 drop-shadow-sm">
        📤 Upload Exercise Video
      </h2>

      <div className="animate-fadeInUp">
        <form
          onSubmit={handleSubmit}
          className="flex flex-col gap-6 text-gray-700 font-medium"
        >
          {/* Exercise Dropdown */}
          <div className="flex flex-col">
            <label className="mb-1 text-gray-800 font-semibold">
              Select Exercise
            </label>
            <select
              className="w-full border border-gray-300 rounded-lg p-3 bg-white/60 focus:ring-2 focus:ring-blue-400 focus:outline-none shadow-sm"
              value={exercise}
              onChange={(e) => setExercise(e.target.value)}
            >
              {exercises.map((ex) => (
                <option key={ex} value={ex}>
                  {ex.charAt(0).toUpperCase() + ex.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* File Upload */}
          <div className="flex flex-col">
            <label className="mb-1 text-gray-800 font-semibold">
              Upload Video File
            </label>
            <div className="relative">
              <input
                type="file"
                accept="video/*"
                className="block w-full border border-gray-300 rounded-lg p-3 pr-12 bg-white/60 focus:ring-2 focus:ring-blue-400 focus:outline-none cursor-pointer shadow-sm"
                onChange={(e) => setFile(e.target.files[0])}
              />
              <span className="absolute right-4 top-3 text-blue-500 text-lg">
                🎥
              </span>
            </div>
            {file && (
              <p className="text-sm text-gray-600 mt-2">
                Selected: <span className="font-semibold">{file.name}</span>
              </p>
            )}
          </div>

          {/* Submit Button */}
          {loading ? (
            <LoadingSpinner />
          ) : (
            <button
              type="submit"
              className="group relative inline-flex items-center justify-center overflow-hidden rounded-full bg-blue-600 px-6 py-3 font-semibold text-white transition-all duration-300 hover:bg-blue-700 hover:shadow-lg"
            >
              <span className="absolute inset-0 bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
              <span className="relative flex items-center gap-2">
                🚀 Analyze
              </span>
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
