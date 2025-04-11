import { db } from './firebase.js';
import {
    collection,
    getDocs,
} from "https://www.gstatic.com/firebasejs/10.11.0/firebase-firestore.js";


document.getElementById("fetchAttendanceBtn").addEventListener("click", async () => {
    const container = document.getElementById("attendanceContainer");
    container.innerHTML = "<p>Fetching data...</p>";
    //   console.log("DB:", db); // Should log Firestore instance

    try {
        const snapshot = await getDocs(collection(db, "loksabha-logs"));
        const allDocs = [];
        snapshot.forEach((doc) => {
            allDocs.push({ id: doc.id, ...doc.data() });
        });
        console.table(allDocs);



        if (snapshot.empty) {
            container.innerHTML = "<p>No attendance records found.</p>";
            return;
        }

        let html = `
          <table class="attendance-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Time</th>
                <th>Date</th>
                <th>Emotion</th>
              </tr>
            </thead>
            <tbody>
        `;

        snapshot.forEach(doc => {
            const data = doc.data();
            html += `
                <tr>
                  <td>${data.name}</td>
                  <td>${data.time}</td>
                  <td>${data.date}</td>
                  <td>${data.Emotion || "—"}</td>
                </tr>
            `;
        });

        html += `
              </tbody>
            </table>
          `;
        container.innerHTML = html;

    } catch (error) {
        console.error("Error fetching Firestore data:", error);
        container.innerHTML = "<p>Error fetching data. Check console.</p>";
    }
});
