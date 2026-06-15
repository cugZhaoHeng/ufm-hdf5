(function () {
  "use strict";

  const API_BASE = window.UFM_API_BASE || "";
  const FACE_LINE_LIMIT = 1200;

  let uploadedH5 = null;
  let uploadedWell = null;
  let elementProperties = [];
  let cellProperties = [];
  let currentTaskId = null;
  let progressTimer = null;
  let resultUrl = null;
  let resultFormat = null;
  let finalStateChart = null;
  let finalStateTimer = null;
  let h5Files = [];
  let wellFiles = [];

  function apiUrl(path) {
    return `${API_BASE}${path}`;
  }

  function showMessage(title, message) {
    if ($.messager) {
      $.messager.alert(title, message);
    } else {
      window.alert(`${title}: ${message}`);
    }
  }

  function getSelectedGroups() {
    return $("#groupsGrid").datagrid("getChecked").map(row => row.group);
  }

  function getSelectedGroupDisplayNames() {
    const result = {};
    $("#groupsGrid").datagrid("getChecked").forEach(row => {
      if (row.group && row.name) {
        result[row.group] = row.name;
      }
    });
    return result;
  }

  function refreshPropertyOptions() {
    const level = $("#levelSelect").combobox("getValue");
    const properties = level === "cell" ? cellProperties : elementProperties;
    const data = properties.map(name => ({ value: name, text: name }));
    $("#propertySelect").combobox("loadData", data);
    if (data.length > 0) {
      $("#propertySelect").combobox("setValue", data[0].value);
    } else {
      $("#propertySelect").combobox("clear");
    }
  }

  function refreshFinalStatePropertyOptions() {
    const level = $("#finalLevelSelect").combobox("getValue") || "element";
    const properties = level === "cell" ? cellProperties : elementProperties;
    const data = properties.map(name => ({ value: name, text: name }));
    $("#finalPropertySelect").combobox("loadData", data);
    if (data.length > 0) {
      $("#finalPropertySelect").combobox("setValue", data[0].value);
    } else {
      $("#finalPropertySelect").combobox("clear");
    }
  }

  function getAllGroups() {
    return (uploadedH5 && uploadedH5.groups ? uploadedH5.groups : []).map(row => row.group);
  }

  function setRunEnabled(enabled) {
    $("#runBtn").linkbutton(enabled ? "enable" : "disable");
  }

  function setPreviewEnabled(enabled) {
    $("#previewBtn").linkbutton(enabled ? "enable" : "disable");
  }

  async function uploadFile(inputId, url) {
    const input = document.getElementById(inputId);
    if (!input.files.length) {
      showMessage("Notice", "Please choose a file first.");
      return null;
    }

    const formData = new FormData();
    formData.append("file", input.files[0]);

    const response = await fetch(apiUrl(url), {
      method: "POST",
      body: formData
    });
    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Upload failed.");
    }
    return data;
  }

  async function uploadH5() {
    try {
      $("#fileInfo").text("Uploading and parsing H5 file...");
      const data = await uploadFile("h5File", "/api/files/upload-h5");
      if (!data) {
        $("#fileInfo").text("No H5 file uploaded.");
        return;
      }

      uploadedH5 = data;
      await loadH5Files(data.file_id);
      applyH5Metadata(data);
    } catch (error) {
      $("#fileInfo").text("Failed to parse H5 file.");
      showMessage("Upload Failed", error.message);
    }
  }

  async function uploadWell() {
    try {
      const data = await uploadFile("wellFile", "/api/files/upload-well");
      if (!data) {
        return;
      }
      uploadedWell = data;
      await loadWellFiles(data.file_id);
      showMessage("Uploaded", `Well trajectory file uploaded: ${data.filename}`);
      scheduleFinalStateRefresh();
    } catch (error) {
      showMessage("Upload Failed", error.message);
    }
  }

  async function loadH5Files(selectedFileId) {
    const response = await fetch(apiUrl("/api/files/h5"));
    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Failed to list H5 files.");
    }
    h5Files = data.files || [];
    $("#h5Select").combobox("loadData", h5Files.map(file => ({
      ...file,
      label: `${file.filename} (${file.source})`
    })));
    if (h5Files.length) {
      $("#h5Select").combobox("setValue", selectedFileId || h5Files[0].file_id);
    }
    $("#fileInfo").text(`Loaded ${h5Files.length} H5 files. Select one and click Parse Selected.`);
  }

  async function loadWellFiles(selectedFileId) {
    const response = await fetch(apiUrl("/api/files/well"));
    const data = await readResponse(response);
    if (!response.ok) {
      throw new Error(data.detail || "Failed to list well files.");
    }
    wellFiles = data.files || [];
    $("#wellSelect").combobox("loadData", wellFiles.map(file => ({
      ...file,
      label: `${file.filename} (${file.source})`
    })));
    if (wellFiles.length) {
      $("#wellSelect").combobox("setValue", selectedFileId || wellFiles[0].file_id);
    }
  }

  async function inspectSelectedH5() {
    const fileId = $("#h5Select").combobox("getValue");
    if (!fileId) {
      showMessage("Notice", "No H5 file is selected.");
      return;
    }
    $("#fileInfo").text("Parsing selected H5 file...");
    try {
      const response = await fetch(apiUrl("/api/files/inspect-h5"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_id: fileId })
      });
      const data = await readResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || "Failed to parse selected H5.");
      }
      applyH5Metadata(data);
    } catch (error) {
      $("#fileInfo").text("Failed to parse selected H5 file.");
      showMessage("Parse Failed", error.message);
    }
  }

  function useSelectedWell() {
    const fileId = $("#wellSelect").combobox("getValue");
    if (!fileId) {
      uploadedWell = null;
      showMessage("Notice", "No well trajectory file is selected.");
      scheduleFinalStateRefresh();
      return;
    }
    const file = wellFiles.find(item => item.file_id === fileId);
    uploadedWell = {
      file_id: fileId,
      filename: file ? file.filename : fileId
    };
    showMessage("Selected", `Well trajectory selected: ${uploadedWell.filename}`);
    scheduleFinalStateRefresh();
  }

  function applyH5Metadata(data) {
    uploadedH5 = data;
    elementProperties = data.element_properties || [];
    cellProperties = data.cell_properties || [];

    $("#groupsGrid").datagrid("loadData", data.groups || []);
    $("#fileInfo").text(`Using ${data.filename}. Parsed ${(data.groups || []).length} groups.`);
    $("#selectAllGroups").prop("checked", false);
    refreshPropertyOptions();
    refreshFinalStatePropertyOptions();
    setRunEnabled(true);
    scheduleFinalStateRefresh();
  }

  async function startAnimation() {
    if (!uploadedH5) {
      showMessage("Notice", "Please upload an H5 file first.");
      return;
    }

    const groupNames = getSelectedGroups();
    if (!groupNames.length) {
      showMessage("Notice", "Please select at least one group.");
      return;
    }

    const propertyName = $("#propertySelect").combobox("getValue");
    if (!propertyName) {
      showMessage("Notice", "Please select a property.");
      return;
    }

    const payload = {
      file_id: uploadedH5.file_id,
      group_names: groupNames,
      group_display_names: getSelectedGroupDisplayNames(),
      level: $("#levelSelect").combobox("getValue"),
      property_name: propertyName,
      renderer: $("#rendererSelect").length ? $("#rendererSelect").combobox("getValue") : "matplotlib",
      output_format: $("#formatSelect").combobox("getValue"),
      well_file_id: uploadedWell ? uploadedWell.file_id : null,
      fps: Number($("#fpsInput").numberspinner("getValue")),
      interval: Number($("#intervalInput").numberspinner("getValue")),
      keep_aspect: $("#animationKeepAspectInput").is(":checked"),
      time_step_stride: Number($("#timeStepStrideInput").length ? $("#timeStepStrideInput").numberspinner("getValue") : 1) || 1
    };
    resultFormat = payload.output_format;
    resultUrl = null;

    $("#downloadBtn").linkbutton("disable");
    setPreviewEnabled(false);
    clearPreview("Generate an animation, then click Preview.");
    $("#progressBar").progressbar("setValue", 0);
    $("#progressText").text("Submitting task...");
    setRunEnabled(false);

    try {
      const response = await fetch(apiUrl("/api/animations"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await readResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || "Failed to create task.");
      }

      currentTaskId = data.task_id;
      pollProgress();
      progressTimer = window.setInterval(pollProgress, 1500);
    } catch (error) {
      setRunEnabled(true);
      showMessage("Start Failed", error.message);
    }
  }

  async function pollProgress() {
    if (!currentTaskId) {
      return;
    }

    const response = await fetch(apiUrl(`/api/tasks/${currentTaskId}`));
    const data = await readResponse(response);
    if (!response.ok) {
      window.clearInterval(progressTimer);
      showMessage("Progress Failed", data.detail || "Unable to read task status.");
      setRunEnabled(true);
      return;
    }

    $("#progressBar").progressbar("setValue", data.percent || 0);
    const groupText = data.current_group ? `Current group: ${data.current_group}` : "No active group.";
    $("#progressText").text(`${data.message || ""} ${groupText} (${data.current_group_index}/${data.total_groups})`);

    if (data.status === "completed") {
      window.clearInterval(progressTimer);
      $("#progressBar").progressbar("setValue", 100);
      $("#progressText").text("Animation completed. Download is ready.");
      $("#downloadBtn").linkbutton("enable");
      resultUrl = apiUrl(data.result_url);
      $("#downloadBtn").attr("href", resultUrl);
      setPreviewEnabled(true);
      setRunEnabled(true);
    }

    if (data.status === "failed") {
      window.clearInterval(progressTimer);
      $("#progressText").text(`Task failed: ${data.error || "unknown error"}`);
      setRunEnabled(true);
    }
  }

  function initGrid() {
    $("#groupsGrid").datagrid({
      data: [],
      fitColumns: false,
      singleSelect: false,
      checkOnSelect: true,
      selectOnCheck: true,
      nowrap: true,
      columns: [[
        { field: "ck", checkbox: true, width: 44 },
        { field: "index", title: "Index", width: 70, align: "center" },
        { field: "group", title: "Group", width: 260 },
        { field: "name", title: "Name", width: 460 },
        { field: "simulation_id", title: "ID", width: 80, align: "center" },
        { field: "time_steps", title: "Steps", width: 90, align: "center" },
        { field: "element_count", title: "Elements", width: 110, align: "center" }
      ]],
      onLoadSuccess: resizeGroupGridColumns,
      onCheck: function () {
        const rows = $("#groupsGrid").datagrid("getRows");
        const checked = $("#groupsGrid").datagrid("getChecked");
        $("#selectAllGroups").prop("checked", rows.length > 0 && rows.length === checked.length);
      },
      onUncheck: function () {
        $("#selectAllGroups").prop("checked", false);
      },
      onCheckAll: function () {
        $("#selectAllGroups").prop("checked", true);
      },
      onUncheckAll: function () {
        $("#selectAllGroups").prop("checked", false);
      }
    });
    resizeGroupGridColumns();
  }

  function resizeGroupGridColumns() {
    const grid = $("#groupsGrid");
    const width = grid.closest(".groups-panel").innerWidth() || grid.width();
    const fixedWidth = 44 + 70 + 80 + 90 + 110 + 48;
    const flexibleWidth = Math.max(width - fixedWidth, 720);
    const groupWidth = Math.max(Math.floor(flexibleWidth * 0.36), 260);
    const nameWidth = Math.max(flexibleWidth - groupWidth, 460);

    grid.datagrid("resize", { width: "100%" });
    grid.datagrid("getColumnOption", "group").width = groupWidth;
    grid.datagrid("getColumnOption", "name").width = nameWidth;
    grid.datagrid("fixColumnSize");
  }

  function clearPreview(message) {
    $("#previewInfo").text("No generated animation yet.");
    $("#previewContainer").html(`<div class="preview-empty">${message}</div>`);
  }

  function showPreview() {
    if (!resultUrl) {
      showMessage("Notice", "No generated animation is ready.");
      return;
    }

    const cacheBustedUrl = `${resultUrl}${resultUrl.includes("?") ? "&" : "?"}t=${Date.now()}`;
    if (resultFormat === "mp4") {
      $("#previewContainer").html(
        `<video src="${cacheBustedUrl}" controls autoplay muted loop></video>`
      );
    } else {
      $("#previewContainer").html(`<img src="${cacheBustedUrl}" alt="Generated animation">`);
    }
    $("#previewInfo").text(resultFormat ? resultFormat.toUpperCase() : "Animation");
  }

  function scheduleFinalStateRefresh() {
    window.clearTimeout(finalStateTimer);
    finalStateTimer = window.setTimeout(loadFinalStateSnapshot, 150);
  }

  async function loadFinalStateSnapshot() {
    if (!uploadedH5) {
      return;
    }
    if (!window.echarts) {
      $("#finalStateInfo").text("ECharts failed to load.");
      return;
    }

    const propertyName = $("#finalPropertySelect").combobox("getValue");
    const level = $("#finalLevelSelect").combobox("getValue") || "element";
    const groupNames = getAllGroups();
    if (!propertyName || !groupNames.length) {
      $("#finalStateInfo").text("No property or group is available.");
      return;
    }

    $("#finalStateInfo").text("Loading final time step state...");
    try {
      const response = await fetch(apiUrl("/api/snapshots/final-state"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_id: uploadedH5.file_id,
          group_names: groupNames,
          level: level,
          property_name: propertyName,
          well_file_id: uploadedWell ? uploadedWell.file_id : null
        })
      });
      const data = await readResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || "Failed to load final state.");
      }
      renderFinalState(data);
    } catch (error) {
      console.error("Final state render failed", error);
      $("#finalStateInfo").text(`Final state failed: ${error.message}`);
    }
  }

  function renderFinalState(data) {
    if (!finalStateChart) {
      finalStateChart = echarts.init(document.getElementById("finalStateChart"));
      window.addEventListener("resize", function () {
        finalStateChart.resize();
      });
    }

    const points = data.points || [];
    const quads = data.quads || [];
    const well = data.well || [];
    const minValue = data.value_min == null ? 0 : data.value_min;
    const maxValue = data.value_max == null ? 1 : data.value_max;
    const bounds = data.bounds && Object.keys(data.bounds).length ? data.bounds : computeBounds(points, well, quads);
    const transformedPoints = transformPointData(points);
    const quadLines = buildQuadLineData(quads, minValue, maxValue);
    const transformedWell = transformWellData(well);
    const series = [];

    if (quadLines.length && quadLines.length <= FACE_LINE_LIMIT) {
      series.push({
        name: `${data.level} ${data.property_name} Values`,
        type: "scatter3D",
        data: transformedPoints,
        symbolSize: 0.1,
        silent: true,
        itemStyle: { opacity: 0 },
        showInLegend: false
      });
      quadLines.forEach((lineItem) => {
        series.push({
          name: `${data.level} ${data.property_name} Faces`,
          type: "line3D",
          coordinateSystem: "cartesian3D",
          showInLegend: false,
          data: lineItem.coords,
          lineStyle: {
            width: data.level === "cell" ? 1 : 1.5,
            opacity: 0.88,
            color: lineItem.color
          }
        });
      });
    } else {
      series.push({
        name: `${data.level} ${data.property_name}`,
        type: "scatter3D",
        data: transformedPoints,
        symbolSize: data.level === "cell" ? 2.5 : 4,
        encode: { x: 0, y: 1, z: 2, tooltip: [0, 1, 2, 3] },
        itemStyle: { opacity: 0.88 }
      });
    }

    if (well.length > 1) {
      series.push({
        name: "Well Trajectory",
        type: "line3D",
        coordinateSystem: "cartesian3D",
        data: transformedWell,
        lineStyle: {
          width: 4,
          color: "#111111"
        }
      });
    }

    finalStateChart.setOption({
      backgroundColor: "#0f1720",
      tooltip: {
        formatter: function (params) {
          const value = params.value || [];
          if (params.seriesName === "Well Trajectory") {
            return `Well<br>X: ${formatNumber(value[0])}<br>Y: ${formatNumber(value[1])}<br>Depth: ${formatNumber(-value[2])}`;
          }
          if (params.seriesType === "line3D") {
            return params.seriesName;
          }
          return [
            params.seriesName,
            `X: ${formatNumber(value[0])}`,
            `Y: ${formatNumber(value[1])}`,
            `Depth: ${formatNumber(-value[2])}`,
            `${data.property_name}: ${formatNumber(value[3])}`,
            `Group: ${value[4] || ""}`
          ].join("<br>");
        }
      },
      visualMap: {
        min: minValue,
        max: maxValue,
        dimension: 3,
        seriesIndex: 0,
        calculable: true,
        orient: "vertical",
        right: 18,
        top: 70,
        textStyle: { color: "#d7dee8" },
        inRange: {
          color: ["#00007f", "#0000ff", "#007fff", "#00ffff", "#7fff7f", "#ffff00", "#ff7f00", "#ff0000", "#7f0000"]
        }
      },
      legend: {
        top: 12,
        textStyle: { color: "#d7dee8" }
      },
      xAxis3D: buildAxis3D("X", bounds.x),
      yAxis3D: buildAxis3D("Y", bounds.y),
      zAxis3D: buildAxis3D("Depth", invertZBound(bounds.z), true),
      grid3D: {
        viewControl: {
          projection: "perspective",
          autoRotate: false
        },
        axisLine: { lineStyle: { color: "#9fb3c8" } },
        axisLabel: { textStyle: { color: "#d7dee8" } },
        splitLine: { lineStyle: { color: "#304156" } },
        ...buildGridBox(bounds, $("#keepAspectInput").is(":checked"))
      },
      series: series
    }, true);

    $("#finalStateInfo").text(
      `Loaded ${quads.length || points.length} ${data.level} ${quadLines.length > FACE_LINE_LIMIT ? "points" : "faces"} from ${data.groups.length} groups` +
      (well.length ? ` with ${well.length} well trajectory points.` : ".")
    );
  }

  function formatNumber(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return "";
    }
    const rounded = Math.round(num * 100) / 100;
    return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(2).replace(/\.?0+$/, "");
  }

  function buildAxis3D(name, bound, invertLabel) {
    const axis = {
      type: "value",
      name: name,
      scale: true,
      axisLabel: {
        formatter: function (value) {
          return formatNumber(invertLabel ? -value : value);
        }
      }
    };
    if (bound && Number.isFinite(Number(bound.min)) && Number.isFinite(Number(bound.max))) {
      axis.min = Number(bound.min);
      axis.max = Number(bound.max);
    }
    return axis;
  }

  function computeBounds(points, well, quads) {
    const xyz = [];
    (points || []).forEach(point => xyz.push(point.slice(0, 3)));
    (well || []).forEach(point => xyz.push(point.slice(0, 3)));
    (quads || []).forEach(quad => {
      quad.slice(0, 4).forEach(point => xyz.push(point));
    });
    const result = {};
    [["x", 0], ["y", 1], ["z", 2]].forEach(([axis, index]) => {
      const values = xyz
        .map(point => Number(point[index]))
        .filter(value => Number.isFinite(value));
      if (!values.length) {
        return;
      }
      const rawMin = Math.min(...values);
      const rawMax = Math.max(...values);
      const span = rawMax - rawMin;
      const pad = Math.max(span * 0.05, 1);
      result[axis] = {
        min: rawMin - pad,
        max: rawMax + pad,
        raw_min: rawMin,
        raw_max: rawMax
      };
    });
    return result;
  }

  function invertZBound(bound) {
    if (!bound) {
      return bound;
    }
    return {
      min: -Number(bound.max),
      max: -Number(bound.min),
      raw_min: -Number(bound.raw_max),
      raw_max: -Number(bound.raw_min)
    };
  }

  function transformPointData(points) {
    return (points || []).map(point => {
      const next = point.slice();
      next[2] = -Number(point[2]);
      return next;
    });
  }

  function transformWellData(well) {
    return (well || []).map(point => [point[0], point[1], -Number(point[2])]);
  }

  function buildQuadLineData(quads, minValue, maxValue) {
    return (quads || []).map(quad => {
      if (!Array.isArray(quad) || quad.length < 5) {
        return null;
      }
      const value = Number(quad[4]);
      const coords = quad.slice(0, 4).map(point => {
        if (!Array.isArray(point) || point.length < 3) {
          return null;
        }
        const x = Number(point[0]);
        const y = Number(point[1]);
        const z = -Number(point[2]);
        return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) ? [x, y, z] : null;
      });
      if (!Number.isFinite(value) || coords.some(point => point == null)) {
        return null;
      }
      coords.push(coords[0]);
      return {
        coords: coords,
        value: value,
        color: jetColor(value, minValue, maxValue)
      };
    }).filter(Boolean);
  }

  function jetColor(value, minValue, maxValue) {
    const min = Number(minValue);
    const max = Number(maxValue);
    const t = Number.isFinite(value) && max > min ? (value - min) / (max - min) : 0.5;
    const clamped = Math.max(0, Math.min(1, t));
    const bucketed = Math.round(clamped * 63) / 63;
    const r = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * bucketed - 3))));
    const g = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * bucketed - 2))));
    const b = Math.round(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * bucketed - 1))));
    return `rgb(${r},${g},${b})`;
  }

  function buildGridBox(bounds, keepAspect) {
    if (!keepAspect || !bounds || !bounds.x || !bounds.y || !bounds.z) {
      return {
        boxWidth: 120,
        boxDepth: 90,
        boxHeight: 70
      };
    }

    const spans = {
      x: Math.max(Number(bounds.x.raw_max) - Number(bounds.x.raw_min), 1),
      y: Math.max(Number(bounds.y.raw_max) - Number(bounds.y.raw_min), 1),
      z: Math.max(Number(bounds.z.raw_max) - Number(bounds.z.raw_min), 1)
    };
    const maxSpan = Math.max(spans.x, spans.y, spans.z, 1);
    const scale = 120 / maxSpan;
    return {
      boxWidth: Math.max(spans.x * scale, 20),
      boxDepth: Math.max(spans.y * scale, 20),
      boxHeight: Math.max(spans.z * scale, 20)
    };
  }

  async function readResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      return response.json();
    }

    const text = await response.text();
    if (response.ok) {
      return { detail: text };
    }

    return {
      detail: `HTTP ${response.status} ${response.statusText}. Check Nginx proxy and upload size settings.`,
      raw: text
    };
  }

  function initPage() {
    if (!window.jQuery || !$.fn.datagrid) {
      document.getElementById("fileInfo").textContent = "EasyUI or jQuery failed to load.";
      return;
    }

    initGrid();
    $("#h5Select").combobox({
      valueField: "file_id",
      textField: "label",
      panelHeight: "auto",
      editable: false
    });
    $("#wellSelect").combobox({
      valueField: "file_id",
      textField: "label",
      panelHeight: "auto",
      editable: false
    });
    loadH5Files().catch(error => {
      $("#fileInfo").text(`Failed to list H5 files: ${error.message}`);
    });
    loadWellFiles().catch(error => {
      showMessage("Well List Failed", error.message);
    });
    $("#uploadH5Btn").on("click", uploadH5);
    $("#inspectH5Btn").on("click", inspectSelectedH5);
    $("#uploadWellBtn").on("click", uploadWell);
    $("#useWellBtn").on("click", useSelectedWell);
    $("#runBtn").on("click", startAnimation);
    $("#previewBtn").on("click", showPreview);
    $("#finalLevelSelect").combobox({
      onChange: function () {
        refreshFinalStatePropertyOptions();
        scheduleFinalStateRefresh();
      }
    });
    $("#finalPropertySelect").combobox({
      onChange: scheduleFinalStateRefresh
    });
    $("#keepAspectInput").on("change", scheduleFinalStateRefresh);
    $("#levelSelect").combobox({
      onChange: function () {
        refreshPropertyOptions();
      }
    });
    $("#selectAllGroups").on("change", function () {
      $("#groupsGrid").datagrid(this.checked ? "checkAll" : "uncheckAll");
    });
    $("#fileInfo").text(`Ready. API base: ${API_BASE || "same origin"}`);
    $(window).on("resize", resizeGroupGridColumns);
  }

  $(initPage);
})();
