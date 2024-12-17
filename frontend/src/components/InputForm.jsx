import React, { useState } from "react";

function InputForm({ onSubmit }) {
  const [grantData, setGrantData] = useState({
    grant_title: "",
    objective: "",
    audience: "",
    funding: "",
    details: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setGrantData({
      ...grantData,
      [name]: value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form submitted with data:", grantData);
    onSubmit(grantData); // Pass data to the parent component
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="grant_title" className="block">Grant Title</label>
        <input
          type="text"
          id="grant_title"
          name="grant_title"
          value={grantData.grant_title}
          onChange={handleChange}
          className="w-full p-2 border"
          required
        />
      </div>
      <div>
        <label htmlFor="objective" className="block">Objective</label>
        <input
          type="text"
          id="objective"
          name="objective"
          value={grantData.objective}
          onChange={handleChange}
          className="w-full p-2 border"
          required
        />
      </div>
      <div>
        <label htmlFor="audience" className="block">Target Audience</label>
        <input
          type="text"
          id="audience"
          name="audience"
          value={grantData.audience}
          onChange={handleChange}
          className="w-full p-2 border"
          required
        />
      </div>
      <div>
        <label htmlFor="funding" className="block">Funding Amount</label>
        <input
          type="text"
          id="funding"
          name="funding"
          value={grantData.funding}
          onChange={handleChange}
          className="w-full p-2 border"
          required
        />
      </div>
      <div>
        <label htmlFor="details" className="block">Project Details</label>
        <textarea
          id="details"
          name="details"
          value={grantData.details}
          onChange={handleChange}
          rows="4"
          className="w-full p-2 border"
          required
        />
      </div>

      <button type="submit" className="w-full bg-green-600 text-white py-2 rounded">
        Submit
      </button>
    </form>
  );
}

export default InputForm;
