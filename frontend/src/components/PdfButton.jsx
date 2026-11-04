// "Download PDF": opens the browser's print dialog with only the report panel (choose "Save as PDF").
// Collapsed sections are opened for the printout and closed again afterwards; the page title becomes the file name.
export default function PdfButton({ title }) {
  const onClick = (e) => {
    const panel = e.currentTarget.closest(".panel");
    const closed = panel ? [...panel.querySelectorAll("details:not([open])")] : [];
    closed.forEach((d) => {
      d.open = true;
    });

    const oldTitle = document.title;
    document.title = `${title} ${new Date().toISOString().slice(0, 10)}`;

    const restore = () => {
      closed.forEach((d) => {
        d.open = false;
      });
      document.title = oldTitle;
      window.removeEventListener("afterprint", restore);
    };
    window.addEventListener("afterprint", restore);
    window.print();
  };

  return (
    <button type="button" className="ghost pdf-button no-print" onClick={onClick}>
      Download PDF
    </button>
  );
}
