import { Stethoscope, User } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Header = () => {
  return (
    <header className="bg-card border-b shadow-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="rounded-lg">
              <img src="/clearRx.png" alt="Logo" className="h-24 w-24" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-foreground">
                clearRx
              </h1>
              <p className="text-sm text-muted-foreground">
                Drug-Drug Interaction Assistant
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm" className="flex items-center space-x-2">
              <User className="h-4 w-4" />
              <span>Dr. Ashwin Vijayakumar</span>
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
};